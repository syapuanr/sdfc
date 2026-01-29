"""
Custom Configuration Example
Contoh menggunakan custom configuration dan presets
"""

import torch
from diffusers import StableDiffusionPipeline
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import DiffusionInferenceEngine
from src.config import RuntimeConfig, PresetConfigs, detect_environment


def example_auto_detect():
    """Example: Auto-detect configuration"""
    print("=" * 70)
    print("Example 1: Auto-Detect Configuration")
    print("=" * 70)
    
    config = detect_environment()
    
    print(f"\n✓ Auto-detected configuration:")
    print(f"  Model: {config.model_id}")
    print(f"  Device: {config.device}")
    print(f"  CPU Offload: {config.enable_cpu_offload}")
    print(f"  Attention Slicing: {config.enable_attention_slicing}")
    print(f"  VAE Tiling: {config.enable_vae_tiling}")
    print(f"  Max Retries: {config.max_retries}")
    
    return config


def example_presets():
    """Example: Using presets"""
    print("\n" + "=" * 70)
    print("Example 2: Using Presets")
    print("=" * 70)
    
    # Low VRAM preset
    print("\n1. Low VRAM Preset (4-6GB):")
    config = PresetConfigs.low_vram()
    print(f"   Model: {config.model_id}")
    print(f"   CPU Offload: {config.enable_cpu_offload}")
    print(f"   All optimizations: ON")
    print(f"   Max retries: {config.max_retries}")
    
    # High VRAM preset
    print("\n2. High VRAM Preset (16GB+):")
    config = PresetConfigs.high_vram()
    print(f"   Model: {config.model_id}")
    print(f"   CPU Offload: {config.enable_cpu_offload}")
    print(f"   Optimizations: Minimal (for speed)")
    
    # Fast inference preset
    print("\n3. Fast Inference Preset:")
    config = PresetConfigs.fast_inference()
    print(f"   CPU Offload: {config.enable_cpu_offload}")
    print(f"   Timeout: {config.timeout_seconds}s")
    print(f"   Optimized for: Speed")
    
    return config


def example_custom_config():
    """Example: Custom configuration"""
    print("\n" + "=" * 70)
    print("Example 3: Custom Configuration")
    print("=" * 70)
    
    config = RuntimeConfig(
        # Model settings
        model_id="stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        
        # Memory management
        enable_cpu_offload=True,
        enable_attention_slicing=True,
        enable_vae_tiling=True,
        
        # Execution settings
        max_retries=5,
        timeout_seconds=600,
        enable_progressive_degradation=True,
        
        # Memory thresholds
        safe_threshold=0.65,
        warning_threshold=0.80,
        critical_threshold=0.92,
        
        # Output
        output_dir=Path("outputs/custom")
    )
    
    print(f"\n✓ Custom configuration created:")
    print(f"  Model: {config.model_id}")
    print(f"  Device: {config.device}")
    print(f"  Max retries: {config.max_retries}")
    print(f"  Timeout: {config.timeout_seconds}s")
    print(f"  Safe threshold: {config.safe_threshold:.0%}")
    print(f"  Output dir: {config.output_dir}")
    
    return config


def example_with_generation():
    """Example: Using config with generation"""
    print("\n" + "=" * 70)
    print("Example 4: Generation with Custom Config")
    print("=" * 70)
    
    # Use low VRAM preset
    config = PresetConfigs.low_vram()
    
    print(f"\nUsing preset: Low VRAM")
    print("Initializing engine...")
    
    engine = DiffusionInferenceEngine(
        model_id=config.model_id,
        device=str(config.device),
        torch_dtype=config.torch_dtype,
        enable_cpu_offload=config.enable_cpu_offload
    )
    
    print("Loading model...")
    engine.initialize(StableDiffusionPipeline)
    
    print("Generating image...")
    result = engine.generate(
        prompt="A futuristic cityscape at night",
        negative_prompt="blurry, low quality",
        num_inference_steps=25,
        height=512,
        width=512
    )
    
    # Save
    output_path = config.output_dir / "custom_config.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)
    
    print(f"\n✓ Image saved to: {output_path}")
    
    # Cleanup
    engine.cleanup()


def example_save_load_config():
    """Example: Save and load configuration"""
    print("\n" + "=" * 70)
    print("Example 5: Save and Load Configuration")
    print("=" * 70)
    
    # Create custom config
    config = RuntimeConfig(
        model_id="runwayml/stable-diffusion-v1-5",
        max_retries=10,
        enable_progressive_degradation=True
    )
    
    # Save to file
    config_path = "my_config.json"
    config.save(config_path)
    print(f"\n✓ Config saved to: {config_path}")
    
    # Load from file
    loaded_config = RuntimeConfig.load(config_path)
    print(f"✓ Config loaded from: {config_path}")
    print(f"  Model: {loaded_config.model_id}")
    print(f"  Max retries: {loaded_config.max_retries}")


def main():
    """Run all examples"""
    
    # Example 1: Auto-detect
    example_auto_detect()
    
    # Example 2: Presets
    example_presets()
    
    # Example 3: Custom config
    example_custom_config()
    
    # Example 4: Save/Load
    example_save_load_config()
    
    # Example 5: Generation (optional - requires model download)
    if torch.cuda.is_available():
        try:
            example_with_generation()
        except Exception as e:
            print(f"\n⚠️  Generation example skipped: {e}")
    else:
        print("\n⚠️  Generation example skipped (no CUDA)")
    
    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
