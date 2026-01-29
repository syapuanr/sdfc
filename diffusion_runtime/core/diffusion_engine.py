"""
Diffusion Inference Engine
Main engine yang mengintegrasikan semua komponen untuk fault-tolerant inference
"""

import torch
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image

from .memory_manager import MemoryManager, MemoryOptimizer
from .model_loader import DiffusionModelLoader, ModelConfig
from .execution_state_machine import (
    ExecutionStateMachine, ExecutionContext, ExecutionState,
    ExecutionMonitor, ErrorType
)
from .job_queue_manager import JobRequest, JobResult, JobStatus, JobManager

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result dari inference"""
    images: List[Image.Image]
    latents: Optional[torch.Tensor] = None
    nsfw_detected: List[bool] = None
    
    # Metadata
    seed: Optional[int] = None
    prompt: str = ""
    negative_prompt: str = ""
    num_steps: int = 0
    guidance_scale: float = 0.0


class DiffusionInferenceEngine:
    """
    Main inference engine dengan fault tolerance dan memory management
    """
    
    def __init__(self,
                 model_id: str,
                 device: str = "cuda",
                 torch_dtype: torch.dtype = torch.float16,
                 enable_cpu_offload: bool = True):
        """
        Args:
            model_id: Model ID (e.g., "runwayml/stable-diffusion-v1-5")
            device: Device untuk inference
            torch_dtype: Data type untuk model weights
            enable_cpu_offload: Enable CPU offloading
        """
        self.device = torch.device(device)
        
        # Initialize components
        self.memory_manager = MemoryManager(device, enable_cpu_offload)
        
        # Model configuration
        self.model_config = ModelConfig(
            model_id=model_id,
            torch_dtype=torch_dtype,
            enable_cpu_offload=enable_cpu_offload
        )
        
        # Model loader
        self.model_loader = DiffusionModelLoader(
            self.model_config,
            self.memory_manager
        )
        
        # State machine
        self.state_machine = ExecutionStateMachine(self.memory_manager)
        self.monitor = ExecutionMonitor(self.state_machine, self.memory_manager)
        
        # Pipeline reference
        self.pipeline = None
        self.is_initialized = False
        
        logger.info(f"DiffusionInferenceEngine initialized for {model_id}")
    
    def initialize(self, pipeline_class):
        """
        Initialize pipeline dan load models
        
        Args:
            pipeline_class: Diffusers pipeline class
        """
        if self.is_initialized:
            logger.warning("Engine already initialized")
            return
        
        logger.info("=" * 60)
        logger.info("INITIALIZING DIFFUSION ENGINE")
        logger.info("=" * 60)
        
        try:
            # Load pipeline
            self.pipeline = self.model_loader.load_pipeline_from_diffusers(
                pipeline_class
            )
            
            # Initial memory report
            report = self.model_loader.get_memory_report()
            logger.info(f"Memory status after initialization: {report['status']}")
            logger.info(
                f"VRAM: {report['vram']['allocated_gb']:.2f}GB / "
                f"{report['vram']['total_gb']:.2f}GB"
            )
            
            self.is_initialized = True
            logger.info("Engine initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            raise
    
    def _encode_prompt(self,
                       prompt: str,
                       negative_prompt: str,
                       num_images: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 1: Encode prompts using text encoder
        
        Args:
            prompt: Main prompt
            negative_prompt: Negative prompt
            num_images: Batch size
        
        Returns:
            (prompt_embeds, negative_prompt_embeds)
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: ENCODING PROMPTS")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Load text encoder ke GPU
            text_encoder = self.model_loader.prepare_text_encoder()
            tokenizer = self.model_loader.tokenizer
            
            # Tokenize prompts
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)
            
            # Encode
            with torch.no_grad():
                prompt_embeds = text_encoder(text_input_ids)[0]
            
            # Negative prompt
            if negative_prompt:
                uncond_inputs = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                uncond_input_ids = uncond_inputs.input_ids.to(self.device)
                
                with torch.no_grad():
                    negative_prompt_embeds = text_encoder(uncond_input_ids)[0]
            else:
                # Use empty embeddings
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            
            # Duplicate untuk batch
            if num_images > 1:
                prompt_embeds = prompt_embeds.repeat(num_images, 1, 1)
                negative_prompt_embeds = negative_prompt_embeds.repeat(num_images, 1, 1)
            
            encoding_time = time.time() - start_time
            context = self.state_machine.current_context
            if context:
                context.metrics.encoding_time = encoding_time
            
            logger.info(f"Prompt encoding completed in {encoding_time:.2f}s")
            logger.info(f"Prompt embeds shape: {prompt_embeds.shape}")
            
            return prompt_embeds, negative_prompt_embeds
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error("OOM during prompt encoding")
            raise
        except Exception as e:
            logger.error(f"Failed to encode prompts: {e}")
            raise
    
    def _run_diffusion(self,
                       prompt_embeds: torch.Tensor,
                       negative_prompt_embeds: torch.Tensor,
                       num_steps: int,
                       guidance_scale: float,
                       height: int,
                       width: int,
                       seed: Optional[int] = None) -> torch.Tensor:
        """
        Phase 2: Run diffusion process using UNet
        
        Returns:
            Latent tensor
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: RUNNING DIFFUSION")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Load UNet ke GPU
            unet = self.model_loader.prepare_unet()
            scheduler = self.model_loader.scheduler
            
            # Set timesteps
            scheduler.set_timesteps(num_steps, device=self.device)
            timesteps = scheduler.timesteps
            
            # Prepare latents
            batch_size = prompt_embeds.shape[0]
            latent_shape = (
                batch_size,
                unet.config.in_channels,
                height // 8,
                width // 8
            )
            
            # Initialize latents with seed
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            latents = torch.randn(
                latent_shape,
                generator=generator,
                device=self.device,
                dtype=prompt_embeds.dtype
            )
            
            # Scale initial latents
            latents = latents * scheduler.init_noise_sigma
            
            # Diffusion loop
            logger.info(f"Running {num_steps} diffusion steps...")
            
            for i, t in enumerate(timesteps):
                # Update progress
                self.monitor.update_progress(i)
                
                # Check health
                if not self.monitor.check_health():
                    raise RuntimeError("Health check failed during diffusion")
                
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                
                # Concatenate embeddings
                text_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                
                # Predict noise
                with torch.no_grad():
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeds,
                    ).sample
                
                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                
                # Compute previous noisy sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            diffusion_time = time.time() - start_time
            context = self.state_machine.current_context
            if context:
                context.metrics.diffusion_time = diffusion_time
            
            logger.info(f"Diffusion completed in {diffusion_time:.2f}s")
            logger.info(f"Latents shape: {latents.shape}")
            
            return latents
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error("OOM during diffusion")
            raise
        except Exception as e:
            logger.error(f"Failed during diffusion: {e}")
            raise
    
    def _decode_latents(self, latents: torch.Tensor) -> List[Image.Image]:
        """
        Phase 3: Decode latents ke images using VAE
        
        Args:
            latents: Latent tensor from diffusion
        
        Returns:
            List of PIL Images
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: DECODING LATENTS")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Load VAE ke GPU
            vae = self.model_loader.prepare_vae()
            
            # Scale latents
            latents = 1 / vae.config.scaling_factor * latents
            
            # Decode
            logger.info(f"Decoding latents with shape {latents.shape}...")
            
            with torch.no_grad():
                images = vae.decode(latents).sample
            
            # Post-process
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).round().astype(np.uint8)
            
            # Convert to PIL
            pil_images = [Image.fromarray(image) for image in images]
            
            decoding_time = time.time() - start_time
            context = self.state_machine.current_context
            if context:
                context.metrics.decoding_time = decoding_time
            
            logger.info(f"Decoding completed in {decoding_time:.2f}s")
            logger.info(f"Generated {len(pil_images)} images")
            
            return pil_images
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error("OOM during VAE decoding")
            
            # Try tiled decoding as fallback
            logger.info("Attempting tiled VAE decoding...")
            try:
                return self._decode_latents_tiled(latents)
            except Exception as e2:
                logger.error(f"Tiled decoding also failed: {e2}")
                raise e
        except Exception as e:
            logger.error(f"Failed to decode latents: {e}")
            raise
    
    def _decode_latents_tiled(self, latents: torch.Tensor) -> List[Image.Image]:
        """
        Fallback: Decode latents dengan tiling untuk reduced memory
        """
        logger.info("Using tiled VAE decoding (memory-efficient mode)")
        
        vae = self.model_loader.vae
        
        # Enable tiling jika belum
        if hasattr(vae, 'enable_tiling'):
            vae.enable_tiling()
        
        # Scale latents
        latents = 1 / vae.config.scaling_factor * latents
        
        # Decode dengan tiling
        with torch.no_grad():
            images = vae.decode(latents).sample
        
        # Post-process
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        
        pil_images = [Image.fromarray(image) for image in images]
        
        logger.info("Tiled decoding successful")
        return pil_images
    
    def generate(self,
                 prompt: str,
                 negative_prompt: str = "",
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 height: int = 512,
                 width: int = 512,
                 num_images: int = 1,
                 seed: Optional[int] = None) -> InferenceResult:
        """
        Main generation method
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale
            height: Image height
            width: Image width
            num_images: Number of images to generate
            seed: Random seed
        
        Returns:
            InferenceResult
        """
        if not self.is_initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        logger.info("=" * 60)
        logger.info("STARTING IMAGE GENERATION")
        logger.info("=" * 60)
        logger.info(f"Prompt: {prompt[:100]}...")
        logger.info(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        logger.info(f"Size: {height}x{width}, Batch: {num_images}")
        
        try:
            # Create execution context
            context = ExecutionContext(
                job_id=f"gen_{int(time.time())}",
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                batch_size=num_images
            )
            
            # Start execution
            self.state_machine.start_execution(context)
            
            # Phase 1: Encode prompts
            self.state_machine._transition_to(ExecutionState.ENCODING_PROMPT)
            prompt_embeds, negative_prompt_embeds = self._encode_prompt(
                prompt, negative_prompt, num_images
            )
            
            # Phase 2: Diffusion
            self.state_machine._transition_to(ExecutionState.DIFFUSION_RUNNING)
            latents = self._run_diffusion(
                prompt_embeds,
                negative_prompt_embeds,
                num_inference_steps,
                guidance_scale,
                height,
                width,
                seed
            )
            
            # Phase 3: Decode
            self.state_machine._transition_to(ExecutionState.DECODING)
            images = self._decode_latents(latents)
            
            # Create result
            result = InferenceResult(
                images=images,
                latents=latents,
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            # Complete execution
            self.state_machine.complete_execution(result)
            self.monitor.finalize_metrics()
            
            # Log final metrics
            status = self.state_machine.get_status()
            metrics = status['context']['metrics']
            logger.info("=" * 60)
            logger.info("GENERATION COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Total time: {metrics['timings']['total']:.2f}s")
            logger.info(f"  Encoding: {metrics['timings']['encoding']:.2f}s")
            logger.info(f"  Diffusion: {metrics['timings']['diffusion']:.2f}s")
            logger.info(f"  Decoding: {metrics['timings']['decoding']:.2f}s")
            logger.info(f"Peak VRAM: {metrics['memory']['peak_vram_gb']:.2f}GB")
            logger.info(f"Retries: {metrics['retries']['total']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            
            # Handle error through state machine
            self.state_machine._handle_error(e)
            
            # Check if we should retry
            context = self.state_machine.current_context
            if context.state == ExecutionState.RETRY_PENDING:
                logger.info("Retrying generation with adjusted parameters...")
                # Recursive retry with adjusted context
                return self.generate(
                    prompt=context.prompt,
                    negative_prompt=context.negative_prompt,
                    num_inference_steps=context.num_steps,
                    guidance_scale=context.guidance_scale,
                    height=context.height,
                    width=context.width,
                    num_images=context.batch_size,
                    seed=seed
                )
            else:
                # Failed permanently
                raise
    
    def cleanup(self):
        """Cleanup engine resources"""
        logger.info("Cleaning up engine...")
        
        if self.model_loader:
            self.model_loader.cleanup()
        
        self.is_initialized = False
        logger.info("Engine cleanup completed")
    
    def get_memory_report(self) -> Dict:
        """Get comprehensive memory report"""
        return self.model_loader.get_memory_report()


if __name__ == "__main__":
    # Example usage
    print("Diffusion Inference Engine Test")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
        
        # This would require actual model download and diffusers library
        print("\nEngine ready for initialization")
        print("To use:")
        print("  from diffusers import StableDiffusionPipeline")
        print("  engine = DiffusionInferenceEngine('runwayml/stable-diffusion-v1-5')")
        print("  engine.initialize(StableDiffusionPipeline)")
        print("  result = engine.generate('A beautiful landscape')")
    else:
        print("CUDA not available")
