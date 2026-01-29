"""
Memory Manager untuk Diffusion Runtime
Menangani monitoring VRAM, garbage collection, dan dynamic offloading
"""

import torch
import gc
import logging
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryLocation(Enum):
    """Lokasi memori untuk model components"""
    GPU = "cuda"
    CPU = "cpu"
    DISK = "disk"


@dataclass
class MemorySnapshot:
    """Snapshot status memori pada waktu tertentu"""
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    total_gb: float
    utilization_percent: float
    timestamp: float


class VRAMMonitor:
    """Monitor VRAM real-time dengan threshold detection"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.total_memory = torch.cuda.get_device_properties(device).total_memory
        self.total_gb = self.total_memory / (1024**3)
        
        # Thresholds
        self.critical_threshold = 0.95  # 95% usage = critical
        self.warning_threshold = 0.85   # 85% usage = warning
        self.safe_threshold = 0.70      # Target 70% untuk operasi aman
        
    def get_memory_stats(self) -> MemorySnapshot:
        """Ambil statistik memori saat ini"""
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        
        allocated_gb = allocated / (1024**3)
        reserved_gb = reserved / (1024**3)
        free_gb = self.total_gb - reserved_gb
        utilization = allocated / self.total_memory
        
        import time
        return MemorySnapshot(
            allocated_gb=allocated_gb,
            reserved_gb=reserved_gb,
            free_gb=free_gb,
            total_gb=self.total_gb,
            utilization_percent=utilization * 100,
            timestamp=time.time()
        )
    
    def is_critical(self) -> bool:
        """Check apakah memori dalam kondisi kritis"""
        stats = self.get_memory_stats()
        return stats.utilization_percent / 100 > self.critical_threshold
    
    def is_warning(self) -> bool:
        """Check apakah memori dalam kondisi warning"""
        stats = self.get_memory_stats()
        return stats.utilization_percent / 100 > self.warning_threshold
    
    def get_available_memory_gb(self) -> float:
        """Dapatkan memori tersedia dalam GB"""
        return self.get_memory_stats().free_gb
    
    def estimate_needed_memory(self, model_size_gb: float, operation: str = "inference") -> float:
        """
        Estimasi memori yang dibutuhkan untuk operasi
        
        Args:
            model_size_gb: Ukuran model dalam GB
            operation: Jenis operasi (inference, training, etc)
        
        Returns:
            Estimated memory in GB
        """
        # Rule of thumb: inference needs ~1.2x model size, dengan overhead
        multipliers = {
            "inference": 1.5,
            "encoding": 1.3,
            "diffusion": 2.0,  # UNet butuh lebih banyak untuk activations
            "decoding": 1.4
        }
        
        multiplier = multipliers.get(operation, 1.5)
        return model_size_gb * multiplier
    
    def can_fit(self, required_gb: float, safety_margin: float = 0.1) -> bool:
        """
        Check apakah operasi bisa fit di memori
        
        Args:
            required_gb: Memori yang dibutuhkan dalam GB
            safety_margin: Safety margin sebagai persentase (0.1 = 10%)
        """
        available = self.get_available_memory_gb()
        required_with_margin = required_gb * (1 + safety_margin)
        return available >= required_with_margin
    
    def log_memory_status(self, prefix: str = ""):
        """Log status memori saat ini"""
        stats = self.get_memory_stats()
        logger.info(
            f"{prefix}VRAM: {stats.allocated_gb:.2f}GB allocated, "
            f"{stats.free_gb:.2f}GB free, "
            f"{stats.utilization_percent:.1f}% used"
        )


class MemoryManager:
    """
    Manager utama untuk memory operations
    Menangani cleanup, offloading, dan memory optimization
    """
    
    def __init__(self, device: str = "cuda:0", enable_cpu_offload: bool = True):
        self.device = torch.device(device)
        self.monitor = VRAMMonitor(device)
        self.enable_cpu_offload = enable_cpu_offload
        
        # Track loaded models
        self.loaded_models: Dict[str, tuple] = {}  # name -> (model, location)
        
        # Memory optimization settings
        self.empty_cache_on_cleanup = True
        self.aggressive_gc = True
        
    def aggressive_cleanup(self):
        """Cleanup agresif untuk membebaskan memori"""
        if self.aggressive_gc:
            gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Aggressive cleanup completed")
        self.monitor.log_memory_status("After cleanup - ")
    
    def register_model(self, name: str, model: torch.nn.Module, location: MemoryLocation):
        """Register model yang di-load untuk tracking"""
        self.loaded_models[name] = (model, location)
        logger.info(f"Registered model '{name}' at {location.value}")
    
    def unregister_model(self, name: str):
        """Unregister dan cleanup model"""
        if name in self.loaded_models:
            model, location = self.loaded_models[name]
            
            # Move to CPU first if on GPU
            if location == MemoryLocation.GPU:
                model.to("cpu")
            
            # Delete reference
            del self.loaded_models[name]
            del model
            
            self.aggressive_cleanup()
            logger.info(f"Unregistered and cleaned up model '{name}'")
    
    def offload_to_cpu(self, model: torch.nn.Module, model_name: str = "model") -> torch.nn.Module:
        """
        Offload model ke CPU
        
        Args:
            model: PyTorch model
            model_name: Nama model untuk logging
        
        Returns:
            Model yang sudah di-offload
        """
        if not self.enable_cpu_offload:
            logger.warning(f"CPU offload disabled, keeping {model_name} on GPU")
            return model
        
        logger.info(f"Offloading {model_name} to CPU...")
        self.monitor.log_memory_status("Before offload - ")
        
        model = model.to("cpu")
        self.aggressive_cleanup()
        
        self.monitor.log_memory_status("After offload - ")
        
        # Update registry if tracked
        if model_name in self.loaded_models:
            self.loaded_models[model_name] = (model, MemoryLocation.CPU)
        
        return model
    
    def load_to_gpu(self, model: torch.nn.Module, model_name: str = "model") -> torch.nn.Module:
        """
        Load model ke GPU dengan safety checks
        
        Args:
            model: PyTorch model
            model_name: Nama model untuk logging
        
        Returns:
            Model yang sudah di-load ke GPU
        """
        logger.info(f"Loading {model_name} to GPU...")
        self.monitor.log_memory_status("Before GPU load - ")
        
        # Estimate memory needed
        param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024**3)
        model_size = param_size + buffer_size
        
        logger.info(f"{model_name} size: {model_size:.2f}GB")
        
        # Check if we can fit
        needed = self.monitor.estimate_needed_memory(model_size, "inference")
        if not self.monitor.can_fit(needed):
            logger.warning(
                f"Insufficient memory for {model_name}. "
                f"Needed: {needed:.2f}GB, Available: {self.monitor.get_available_memory_gb():.2f}GB"
            )
            # Try aggressive cleanup first
            self.aggressive_cleanup()
            
            if not self.monitor.can_fit(needed):
                raise MemoryError(
                    f"Cannot fit {model_name} in GPU memory even after cleanup. "
                    f"Consider using smaller model or enabling tiling."
                )
        
        # Load to GPU
        model = model.to(self.device)
        
        self.monitor.log_memory_status("After GPU load - ")
        
        # Update registry if tracked
        if model_name in self.loaded_models:
            self.loaded_models[model_name] = (model, MemoryLocation.GPU)
        
        return model
    
    def ensure_gpu_space(self, required_gb: float, exclude_models: List[str] = None):
        """
        Pastikan ada cukup ruang GPU dengan offloading jika perlu
        
        Args:
            required_gb: Memori yang dibutuhkan dalam GB
            exclude_models: Model yang tidak boleh di-offload
        """
        exclude_models = exclude_models or []
        
        while not self.monitor.can_fit(required_gb):
            # Find models to offload (yang tidak di-exclude)
            gpu_models = [
                (name, model) for name, (model, loc) in self.loaded_models.items()
                if loc == MemoryLocation.GPU and name not in exclude_models
            ]
            
            if not gpu_models:
                # No more models to offload
                self.aggressive_cleanup()
                if not self.monitor.can_fit(required_gb):
                    raise MemoryError(
                        f"Cannot free enough GPU memory. "
                        f"Needed: {required_gb:.2f}GB, "
                        f"Available: {self.monitor.get_available_memory_gb():.2f}GB"
                    )
                break
            
            # Offload oldest/first model
            name, model = gpu_models[0]
            logger.info(f"Offloading {name} to free GPU space...")
            self.offload_to_cpu(model, name)
    
    def get_memory_report(self) -> Dict:
        """Generate comprehensive memory report"""
        stats = self.monitor.get_memory_stats()
        
        report = {
            "vram": {
                "total_gb": stats.total_gb,
                "allocated_gb": stats.allocated_gb,
                "free_gb": stats.free_gb,
                "utilization_percent": stats.utilization_percent
            },
            "loaded_models": {
                name: location.value 
                for name, (_, location) in self.loaded_models.items()
            },
            "status": "CRITICAL" if self.monitor.is_critical() 
                     else "WARNING" if self.monitor.is_warning() 
                     else "OK"
        }
        
        return report


class MemoryOptimizer:
    """
    Optimizer untuk operasi-operasi yang memory-intensive
    Implements tiling, chunking, dan attention optimization
    """
    
    @staticmethod
    def enable_attention_slicing(unet: torch.nn.Module, slice_size: int = 1):
        """Enable attention slicing untuk mengurangi memory usage"""
        if hasattr(unet, 'set_attention_slice'):
            unet.set_attention_slice(slice_size)
            logger.info(f"Enabled attention slicing with size {slice_size}")
    
    @staticmethod
    def enable_vae_slicing(vae: torch.nn.Module):
        """Enable VAE slicing untuk tiled decoding"""
        if hasattr(vae, 'enable_slicing'):
            vae.enable_slicing()
            logger.info("Enabled VAE slicing")
    
    @staticmethod
    def enable_vae_tiling(vae: torch.nn.Module, tile_sample_min_size: int = 512):
        """Enable VAE tiling untuk large images"""
        if hasattr(vae, 'enable_tiling'):
            vae.enable_tiling()
            logger.info(f"Enabled VAE tiling with min size {tile_sample_min_size}")
    
    @staticmethod
    def calculate_optimal_tile_size(image_size: tuple, vram_available_gb: float) -> tuple:
        """
        Calculate optimal tile size untuk VAE decoding
        
        Args:
            image_size: (height, width) target image
            vram_available_gb: Available VRAM in GB
        
        Returns:
            (tile_height, tile_width)
        """
        h, w = image_size
        
        # Base tile size pada available VRAM
        # Rule of thumb: 512x512 needs ~1GB, scale accordingly
        base_tile = 512
        if vram_available_gb >= 8:
            tile_size = 1024
        elif vram_available_gb >= 4:
            tile_size = 768
        elif vram_available_gb >= 2:
            tile_size = 512
        else:
            tile_size = 384
        
        # Ensure tiles don't exceed image size
        tile_h = min(tile_size, h)
        tile_w = min(tile_size, w)
        
        # Make sure divisible by 8 (VAE requirement)
        tile_h = (tile_h // 8) * 8
        tile_w = (tile_w // 8) * 8
        
        logger.info(f"Optimal tile size for {h}x{w} image: {tile_h}x{tile_w}")
        return (tile_h, tile_w)
    
    @staticmethod
    def enable_memory_efficient_attention(model: torch.nn.Module, backend: str = "xformers"):
        """
        Enable memory efficient attention (xformers, flash-attention, etc)
        
        Args:
            model: Model to optimize
            backend: Backend to use (xformers, flash, sdp)
        """
        try:
            if backend == "xformers":
                if hasattr(model, 'enable_xformers_memory_efficient_attention'):
                    model.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
            elif backend == "flash":
                # Flash attention 2.0
                logger.info("Flash attention requires model compilation support")
            elif backend == "sdp":
                # Scaled dot product attention (PyTorch 2.0+)
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    logger.info("Using PyTorch SDP attention (enabled by default in 2.0+)")
        except Exception as e:
            logger.warning(f"Could not enable {backend} attention: {e}")


if __name__ == "__main__":
    # Example usage
    print("Memory Manager Test")
    print("=" * 50)
    
    if torch.cuda.is_available():
        manager = MemoryManager()
        
        # Show initial state
        print("\nInitial memory state:")
        report = manager.get_memory_report()
        print(f"VRAM: {report['vram']['allocated_gb']:.2f}GB / {report['vram']['total_gb']:.2f}GB")
        print(f"Status: {report['status']}")
        
        # Test cleanup
        print("\nTesting aggressive cleanup...")
        manager.aggressive_cleanup()
        
        # Test memory monitoring
        print("\nMemory monitoring test:")
        manager.monitor.log_memory_status("Current - ")
        
        print(f"\nCan fit 2GB? {manager.monitor.can_fit(2.0)}")
        print(f"Available memory: {manager.monitor.get_available_memory_gb():.2f}GB")
    else:
        print("CUDA not available, skipping tests")
