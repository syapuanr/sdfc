"""
Basic Generation Example
Contoh paling sederhana untuk generate image dengan Diffusion Runtime
"""

import torch
from diffusers import StableDiffusionPipeline
import sys
from pathlib import Path

# Add parent directory to path untuk import
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import DiffusionInferenceEngine
from src.config import RuntimeConfig


def main():
    """Basic single image generation"""
    print("=" * 70)
    print("Basic Image Generation Example")
    print("=" * 70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Running on CPU (very slow).")
        device = "cpu"
    else:
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    
    # Create engine
    print("\nInitializing engine...")
    engine = DiffusionInferenceEngine(
        model_id="runwayml/stable-diffusion-v1-5",
        device=device,
        torch_dtype=torch.float16,
        enable_cpu_offload=True
    )
    
    # Initialize
    print("Loading model...")
    engine.initialize(StableDiffusionPipeline)
    
    # Generate
    print("\nGenerating image...")
    result = engine.generate(
        prompt="A serene landscape with mountains and lake at sunset",
        negative_prompt="blurry, low quality",
        num_inference_steps=30,
        height=512,
        width=512,
        seed=42
    )
    
    # Save
    output_path = Path("outputs") / "basic_generation.png"
    output_path.parent.mkdir(exist_ok=True)
    result.images[0].save(output_path)
    
    print(f"\n✓ Image saved to: {output_path}")
    print(f"✓ Time: {engine.state_machine.current_context.metrics.total_time:.2f}s")
    
    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    main()
