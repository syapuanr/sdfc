"""
Configuration file untuk Diffusion Runtime
Centralized configuration untuk easy customization
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import torch


@dataclass
class RuntimeConfig:
    """Main runtime configuration"""
    
    # Model settings
    model_id: str = "runwayml/stable-diffusion-v1-5"
    variant: Optional[str] = "fp16"
    torch_dtype: torch.dtype = torch.float16
    device: str = "cuda"
    
    # Memory management
    enable_cpu_offload: bool = True
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_xformers: bool = True
    
    # Memory thresholds (percentages)
    safe_threshold: float = 0.70      # Target 70% for safe operation
    warning_threshold: float = 0.85   # 85% triggers warnings
    critical_threshold: float = 0.95  # 95% triggers aggressive cleanup
    
    # Job queue settings
    max_queue_size: int = 100
    num_workers: int = 1  # Usually 1 for single GPU
    
    # Execution settings
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    timeout_seconds: Optional[float] = 300  # 5 minutes
    
    # Progressive degradation
    enable_progressive_degradation: bool = True
    reduce_batch_size_on_oom: bool = True
    enable_tiling_on_oom: bool = True
    reduce_resolution_on_oom: bool = False  # Last resort
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "diffusion_runtime.log"
    enable_memory_logging: bool = True
    log_interval_steps: int = 10  # Log every N steps
    
    # Output settings
    output_dir: Path = Path("outputs")
    save_intermediate: bool = False
    intermediate_save_interval: int = 10  # Save every N steps
    
    # Safety
    enable_safety_checker: bool = True
    
    # Performance
    use_safetensors: bool = True
    low_cpu_mem_usage: bool = True
    
    def __post_init__(self):
        """Validate and setup configuration"""
        # Convert string device to torch.device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate thresholds
        assert 0 < self.safe_threshold < self.warning_threshold < self.critical_threshold <= 1.0
        
        # Check CUDA availability if cuda device specified
        if "cuda" in str(self.device) and not torch.cuda.is_available():
            print(f"WARNING: CUDA device specified but CUDA not available")
            print(f"Falling back to CPU")
            self.device = torch.device("cpu")
            self.enable_cpu_offload = False  # No offloading needed on CPU
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            elif isinstance(value, torch.dtype):
                config_dict[key] = str(value)
            elif isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RuntimeConfig':
        """Create from dictionary"""
        # Convert string types back
        if 'torch_dtype' in config_dict:
            dtype_str = config_dict['torch_dtype']
            if 'float16' in dtype_str:
                config_dict['torch_dtype'] = torch.float16
            elif 'float32' in dtype_str:
                config_dict['torch_dtype'] = torch.float32
        
        if 'output_dir' in config_dict:
            config_dict['output_dir'] = Path(config_dict['output_dir'])
        
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'RuntimeConfig':
        """Load configuration from JSON file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Preset configurations untuk different scenarios

class PresetConfigs:
    """Preset configurations untuk common scenarios"""
    
    @staticmethod
    def colab_free() -> RuntimeConfig:
        """Google Colab free tier (~12GB VRAM T4)"""
        return RuntimeConfig(
            model_id="runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            enable_cpu_offload=True,
            enable_attention_slicing=True,
            enable_vae_tiling=True,
            enable_xformers=True,
            max_retries=3,
            enable_progressive_degradation=True
        )
    
    @staticmethod
    def low_vram() -> RuntimeConfig:
        """Low VRAM systems (4-6GB)"""
        return RuntimeConfig(
            model_id="runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            enable_cpu_offload=True,
            enable_attention_slicing=True,
            enable_vae_slicing=True,
            enable_vae_tiling=True,
            enable_xformers=True,
            max_retries=5,  # More retries
            enable_progressive_degradation=True,
            reduce_batch_size_on_oom=True,
            enable_tiling_on_oom=True,
            reduce_resolution_on_oom=True,  # Enable last resort
            safe_threshold=0.60,  # More conservative
            warning_threshold=0.75,
            critical_threshold=0.90
        )
    
    @staticmethod
    def high_vram() -> RuntimeConfig:
        """High VRAM systems (16GB+)"""
        return RuntimeConfig(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            enable_cpu_offload=False,  # Keep everything on GPU
            enable_attention_slicing=False,
            enable_vae_tiling=False,
            enable_xformers=True,
            max_retries=2,  # Less likely to need retries
            enable_progressive_degradation=False,
            safe_threshold=0.80,  # Less conservative
            warning_threshold=0.90,
            critical_threshold=0.95
        )
    
    @staticmethod
    def fast_inference() -> RuntimeConfig:
        """Optimized for speed (may use more memory)"""
        return RuntimeConfig(
            model_id="runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            enable_cpu_offload=False,
            enable_attention_slicing=False,  # Faster but more memory
            enable_vae_slicing=False,
            enable_vae_tiling=False,
            enable_xformers=True,
            max_retries=1,  # Fail fast
            timeout_seconds=60,  # Shorter timeout
        )
    
    @staticmethod
    def quality_focused() -> RuntimeConfig:
        """Optimized for quality (slower, may need more VRAM)"""
        return RuntimeConfig(
            model_id="stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float32,  # Full precision
            enable_cpu_offload=True,
            enable_attention_slicing=False,
            enable_vae_slicing=False,
            enable_vae_tiling=False,
            max_retries=5,
            timeout_seconds=600,  # Longer timeout
            enable_progressive_degradation=False,  # Don't degrade quality
        )
    
    @staticmethod
    def batch_processing() -> RuntimeConfig:
        """Optimized for batch processing many jobs"""
        return RuntimeConfig(
            model_id="runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            enable_cpu_offload=True,
            max_queue_size=1000,  # Large queue
            max_retries=2,  # Don't spend too much time on failures
            save_intermediate=False,  # Don't slow down with saves
            log_interval_steps=50,  # Less frequent logging
        )


# Environment detection

def detect_environment() -> RuntimeConfig:
    """
    Auto-detect environment dan return appropriate config
    """
    if not torch.cuda.is_available():
        print("No CUDA available, using CPU config (very slow!)")
        return RuntimeConfig(device="cpu", enable_cpu_offload=False)
    
    # Get VRAM
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0).lower()
    
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {vram_gb:.2f}GB")
    
    # Detect Colab
    try:
        import google.colab
        in_colab = True
    except:
        in_colab = False
    
    if in_colab:
        print("Detected Google Colab environment")
        if vram_gb < 8:
            print("Using low VRAM config")
            return PresetConfigs.low_vram()
        else:
            print("Using Colab free tier config")
            return PresetConfigs.colab_free()
    
    # Local environment
    if vram_gb < 6:
        print("Low VRAM detected, using conservative config")
        return PresetConfigs.low_vram()
    elif vram_gb >= 16:
        print("High VRAM detected, using high performance config")
        return PresetConfigs.high_vram()
    else:
        print("Medium VRAM detected, using balanced config")
        return PresetConfigs.colab_free()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Runtime Configuration Examples")
    print("=" * 70)
    
    # Auto-detect
    print("\n1. Auto-detected configuration:")
    config = detect_environment()
    print(f"Model: {config.model_id}")
    print(f"Device: {config.device}")
    print(f"CPU Offload: {config.enable_cpu_offload}")
    print(f"Memory optimizations enabled: {config.enable_vae_tiling}")
    
    # Manual preset
    print("\n2. Using low VRAM preset:")
    config = PresetConfigs.low_vram()
    print(f"Model: {config.model_id}")
    print(f"Max retries: {config.max_retries}")
    print(f"Safety thresholds: {config.safe_threshold:.0%}/{config.warning_threshold:.0%}/{config.critical_threshold:.0%}")
    
    # Save and load
    print("\n3. Saving and loading config:")
    config.save("runtime_config.json")
    print("Config saved to runtime_config.json")
    
    loaded_config = RuntimeConfig.load("runtime_config.json")
    print("Config loaded successfully")
    print(f"Loaded model: {loaded_config.model_id}")
    
    # Custom config
    print("\n4. Custom configuration:")
    custom_config = RuntimeConfig(
        model_id="stabilityai/stable-diffusion-2-1",
        max_retries=5,
        timeout_seconds=600,
        output_dir=Path("my_outputs")
    )
    print(f"Model: {custom_config.model_id}")
    print(f"Output dir: {custom_config.output_dir}")
    print(f"Timeout: {custom_config.timeout_seconds}s")
