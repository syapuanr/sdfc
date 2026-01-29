"""
Model Loader untuk Diffusion Runtime
Menangani phase-based loading: Text Encoder → UNet → VAE
dengan dynamic offloading dan lazy loading
"""

import torch
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from memory_manager import MemoryManager, MemoryLocation, MemoryOptimizer

logger = logging.getLogger(__name__)


class ModelComponent(Enum):
    """Komponen model difusi"""
    TEXT_ENCODER = "text_encoder"
    TEXT_ENCODER_2 = "text_encoder_2"  # Untuk SDXL
    UNET = "unet"
    VAE = "vae"
    SAFETY_CHECKER = "safety_checker"
    TOKENIZER = "tokenizer"


@dataclass
class ModelConfig:
    """Konfigurasi untuk model loading"""
    model_id: str
    variant: Optional[str] = None  # fp16, fp32, etc
    torch_dtype: torch.dtype = torch.float16
    device_map: Optional[str] = None
    low_cpu_mem_usage: bool = True
    use_safetensors: bool = True
    
    # Memory optimization flags
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_xformers: bool = True
    
    # Offload settings
    enable_cpu_offload: bool = True
    enable_sequential_cpu_offload: bool = False


class ComponentLoader:
    """Base loader untuk individual components"""
    
    def __init__(self, memory_manager: MemoryManager, config: ModelConfig):
        self.memory_manager = memory_manager
        self.config = config
        self.component_cache: Dict[str, torch.nn.Module] = {}
    
    def load_from_pretrained(self, 
                            component_type: ModelComponent,
                            load_fn: Callable,
                            **kwargs) -> torch.nn.Module:
        """
        Generic loader untuk component dari pretrained
        
        Args:
            component_type: Tipe component
            load_fn: Function untuk load (e.g., CLIPTextModel.from_pretrained)
            **kwargs: Additional arguments untuk load_fn
        
        Returns:
            Loaded component
        """
        component_name = component_type.value
        
        # Check cache first
        if component_name in self.component_cache:
            logger.info(f"{component_name} already in cache")
            return self.component_cache[component_name]
        
        logger.info(f"Loading {component_name} from {self.config.model_id}")
        
        # Prepare load arguments
        load_kwargs = {
            "pretrained_model_name_or_path": self.config.model_id,
            "torch_dtype": self.config.torch_dtype,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            "use_safetensors": self.config.use_safetensors,
        }
        
        if self.config.variant:
            load_kwargs["variant"] = self.config.variant
        
        # Merge with custom kwargs
        load_kwargs.update(kwargs)
        
        # Load component
        try:
            component = load_fn(**load_kwargs)
            component.eval()  # Set to eval mode
            
            # Disable gradient computation
            for param in component.parameters():
                param.requires_grad = False
            
            # Cache
            self.component_cache[component_name] = component
            
            # Register with memory manager
            self.memory_manager.register_model(
                component_name, 
                component, 
                MemoryLocation.CPU
            )
            
            logger.info(f"Successfully loaded {component_name}")
            return component
            
        except Exception as e:
            logger.error(f"Failed to load {component_name}: {e}")
            raise
    
    def unload_component(self, component_type: ModelComponent):
        """Unload component dan cleanup memory"""
        component_name = component_type.value
        
        if component_name in self.component_cache:
            self.memory_manager.unregister_model(component_name)
            del self.component_cache[component_name]
            logger.info(f"Unloaded {component_name}")


class DiffusionModelLoader:
    """
    Main loader untuk complete diffusion pipeline
    Implements phase-based loading strategy
    """
    
    def __init__(self, 
                 config: ModelConfig,
                 memory_manager: Optional[MemoryManager] = None):
        self.config = config
        self.memory_manager = memory_manager or MemoryManager()
        self.component_loader = ComponentLoader(self.memory_manager, config)
        
        # Pipeline components
        self.text_encoder: Optional[torch.nn.Module] = None
        self.text_encoder_2: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.tokenizer_2: Optional[Any] = None
        self.unet: Optional[torch.nn.Module] = None
        self.vae: Optional[torch.nn.Module] = None
        self.scheduler: Optional[Any] = None
        
        # Track loading state
        self.components_loaded = set()
        
        logger.info(f"Initialized DiffusionModelLoader for {config.model_id}")
    
    def load_pipeline_from_diffusers(self, pipeline_class):
        """
        Load complete pipeline using diffusers library
        
        Args:
            pipeline_class: Diffusers pipeline class (e.g., StableDiffusionPipeline)
        """
        logger.info("Loading complete pipeline from diffusers...")
        
        try:
            # Load pipeline dengan optimizations
            pipeline = pipeline_class.from_pretrained(
                self.config.model_id,
                torch_dtype=self.config.torch_dtype,
                variant=self.config.variant,
                use_safetensors=self.config.use_safetensors,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
            )
            
            # Extract components
            self.text_encoder = pipeline.text_encoder
            self.tokenizer = pipeline.tokenizer
            self.unet = pipeline.unet
            self.vae = pipeline.vae
            self.scheduler = pipeline.scheduler
            
            # Handle SDXL (dual text encoders)
            if hasattr(pipeline, 'text_encoder_2'):
                self.text_encoder_2 = pipeline.text_encoder_2
                self.tokenizer_2 = pipeline.tokenizer_2
            
            # Apply optimizations
            self._apply_optimizations()
            
            # Initial offload - keep everything on CPU
            self._offload_all_to_cpu()
            
            logger.info("Pipeline loaded successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
    
    def _apply_optimizations(self):
        """Apply memory optimizations ke semua components"""
        logger.info("Applying memory optimizations...")
        
        # Attention optimizations
        if self.config.enable_attention_slicing and self.unet:
            MemoryOptimizer.enable_attention_slicing(self.unet)
        
        # VAE optimizations
        if self.vae:
            if self.config.enable_vae_slicing:
                MemoryOptimizer.enable_vae_slicing(self.vae)
            if self.config.enable_vae_tiling:
                MemoryOptimizer.enable_vae_tiling(self.vae)
        
        # XFormers
        if self.config.enable_xformers and self.unet:
            MemoryOptimizer.enable_memory_efficient_attention(self.unet, "xformers")
    
    def _offload_all_to_cpu(self):
        """Offload semua components ke CPU"""
        logger.info("Offloading all components to CPU...")
        
        if self.text_encoder:
            self.text_encoder = self.memory_manager.offload_to_cpu(
                self.text_encoder, "text_encoder"
            )
        
        if self.text_encoder_2:
            self.text_encoder_2 = self.memory_manager.offload_to_cpu(
                self.text_encoder_2, "text_encoder_2"
            )
        
        if self.unet:
            self.unet = self.memory_manager.offload_to_cpu(
                self.unet, "unet"
            )
        
        if self.vae:
            self.vae = self.memory_manager.offload_to_cpu(
                self.vae, "vae"
            )
    
    def prepare_text_encoder(self) -> torch.nn.Module:
        """
        Phase 1: Load text encoder ke GPU
        Offload components lain jika perlu
        """
        logger.info("=" * 50)
        logger.info("PHASE 1: Preparing Text Encoder")
        logger.info("=" * 50)
        
        if not self.text_encoder:
            raise RuntimeError("Text encoder not loaded. Call load_pipeline_from_diffusers first.")
        
        # Offload unet dan vae jika ada di GPU
        if self.unet:
            self.unet = self.memory_manager.offload_to_cpu(self.unet, "unet")
        if self.vae:
            self.vae = self.memory_manager.offload_to_cpu(self.vae, "vae")
        
        # Load text encoder ke GPU
        self.text_encoder = self.memory_manager.load_to_gpu(
            self.text_encoder, "text_encoder"
        )
        
        # Handle second text encoder (SDXL)
        if self.text_encoder_2:
            self.text_encoder_2 = self.memory_manager.load_to_gpu(
                self.text_encoder_2, "text_encoder_2"
            )
        
        self.components_loaded.add(ModelComponent.TEXT_ENCODER)
        logger.info("Text encoder ready on GPU")
        
        return self.text_encoder
    
    def prepare_unet(self) -> torch.nn.Module:
        """
        Phase 2: Load UNet ke GPU
        Offload text encoder dan VAE
        """
        logger.info("=" * 50)
        logger.info("PHASE 2: Preparing UNet")
        logger.info("=" * 50)
        
        if not self.unet:
            raise RuntimeError("UNet not loaded. Call load_pipeline_from_diffusers first.")
        
        # Offload text encoders
        if self.text_encoder:
            self.text_encoder = self.memory_manager.offload_to_cpu(
                self.text_encoder, "text_encoder"
            )
        if self.text_encoder_2:
            self.text_encoder_2 = self.memory_manager.offload_to_cpu(
                self.text_encoder_2, "text_encoder_2"
            )
        
        # Offload VAE jika ada
        if self.vae:
            self.vae = self.memory_manager.offload_to_cpu(self.vae, "vae")
        
        # Load UNet ke GPU
        self.unet = self.memory_manager.load_to_gpu(self.unet, "unet")
        
        self.components_loaded.add(ModelComponent.UNET)
        logger.info("UNet ready on GPU")
        
        return self.unet
    
    def prepare_vae(self) -> torch.nn.Module:
        """
        Phase 3: Load VAE ke GPU
        Offload UNet dan text encoder
        """
        logger.info("=" * 50)
        logger.info("PHASE 3: Preparing VAE")
        logger.info("=" * 50)
        
        if not self.vae:
            raise RuntimeError("VAE not loaded. Call load_pipeline_from_diffusers first.")
        
        # Offload UNet
        if self.unet:
            self.unet = self.memory_manager.offload_to_cpu(self.unet, "unet")
        
        # Offload text encoders
        if self.text_encoder:
            self.text_encoder = self.memory_manager.offload_to_cpu(
                self.text_encoder, "text_encoder"
            )
        if self.text_encoder_2:
            self.text_encoder_2 = self.memory_manager.offload_to_cpu(
                self.text_encoder_2, "text_encoder_2"
            )
        
        # Load VAE ke GPU
        self.vae = self.memory_manager.load_to_gpu(self.vae, "vae")
        
        self.components_loaded.add(ModelComponent.VAE)
        logger.info("VAE ready on GPU")
        
        return self.vae
    
    def get_memory_report(self) -> Dict:
        """Get comprehensive memory report"""
        report = self.memory_manager.get_memory_report()
        report["loaded_components"] = [c.value for c in self.components_loaded]
        return report
    
    def cleanup(self):
        """Cleanup semua resources"""
        logger.info("Cleaning up model loader...")
        
        # Offload everything
        self._offload_all_to_cpu()
        
        # Clear references
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.unet = None
        self.vae = None
        self.scheduler = None
        
        self.components_loaded.clear()
        
        # Aggressive cleanup
        self.memory_manager.aggressive_cleanup()
        
        logger.info("Cleanup completed")


class LazyModelLoader(DiffusionModelLoader):
    """
    Extended loader dengan lazy loading support
    Components hanya di-load saat dibutuhkan
    """
    
    def __init__(self, config: ModelConfig, memory_manager: Optional[MemoryManager] = None):
        super().__init__(config, memory_manager)
        self._pipeline_loaded = False
    
    def ensure_component_loaded(self, component: ModelComponent):
        """
        Ensure component sudah di-load sebelum digunakan
        Lazy load jika belum
        """
        if not self._pipeline_loaded:
            raise RuntimeError(
                "Pipeline not loaded. Call load_pipeline_from_diffusers first."
            )
        
        if component not in self.components_loaded:
            if component == ModelComponent.TEXT_ENCODER:
                self.prepare_text_encoder()
            elif component == ModelComponent.UNET:
                self.prepare_unet()
            elif component == ModelComponent.VAE:
                self.prepare_vae()


if __name__ == "__main__":
    # Example usage
    print("Model Loader Test")
    print("=" * 50)
    
    if torch.cuda.is_available():
        config = ModelConfig(
            model_id="runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            enable_cpu_offload=True
        )
        
        memory_manager = MemoryManager()
        loader = DiffusionModelLoader(config, memory_manager)
        
        print("\nInitial memory state:")
        report = loader.get_memory_report()
        print(f"Status: {report['status']}")
        print(f"VRAM: {report['vram']['allocated_gb']:.2f}GB / {report['vram']['total_gb']:.2f}GB")
        
        print("\nLoader ready for pipeline loading")
        print("Components can be loaded on-demand with:")
        print("  - loader.prepare_text_encoder()")
        print("  - loader.prepare_unet()")
        print("  - loader.prepare_vae()")
    else:
        print("CUDA not available, skipping tests")
