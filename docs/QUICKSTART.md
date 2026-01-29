# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install xformers for better memory efficiency
pip install xformers
```

## Basic Usage

### 1. Simple Generation

```python
import torch
from diffusers import StableDiffusionPipeline
from example_usage import DiffusionRuntime

# Create and start runtime
runtime = DiffusionRuntime(
    model_id="runwayml/stable-diffusion-v1-5",
    device="cuda"
)
runtime.start(StableDiffusionPipeline)

# Generate image
result = runtime.generate_sync(
    prompt="A beautiful sunset over mountains",
    num_steps=30
)

# Save result
result.result.images[0].save("output.png")

# Clean up
runtime.stop()
```

### 2. With Auto-Config

```python
from config import detect_environment
from example_usage import DiffusionRuntime

# Auto-detect best configuration
config = detect_environment()

# Use in runtime (with custom parameters)
runtime = DiffusionRuntime(
    model_id=config.model_id,
    device=str(config.device)
)
```

### 3. Batch Processing

```python
runtime.start(StableDiffusionPipeline)

# Submit multiple jobs
prompts = [
    "A cat on a table",
    "A dog in a park",
    "A bird in the sky"
]

job_ids = []
for prompt in prompts:
    job_id = runtime.submit_job(
        prompt=prompt,
        num_steps=25
    )
    job_ids.append(job_id)

# Wait and collect results
import time
while True:
    stats = runtime.get_stats()
    if stats['queue']['queue_size'] == 0:
        break
    time.sleep(1)

# Get results
for i, job_id in enumerate(job_ids):
    result = runtime.get_job_result(job_id)
    if result.result:
        result.result.images[0].save(f"output_{i}.png")

runtime.stop()
```

## Configuration Presets

```python
from config import PresetConfigs

# For Google Colab free tier
config = PresetConfigs.colab_free()

# For low VRAM systems (4-6GB)
config = PresetConfigs.low_vram()

# For high VRAM systems (16GB+)
config = PresetConfigs.high_vram()

# For speed-optimized inference
config = PresetConfigs.fast_inference()

# Save custom config
config.save("my_config.json")

# Load config
from config import RuntimeConfig
config = RuntimeConfig.load("my_config.json")
```

## Monitoring and Debugging

```python
from utils import MemoryMonitor, DebugLogger

# Monitor memory during inference
monitor = MemoryMonitor()
monitor.start()

# Run inference...

stats = monitor.stop()
print(f"Peak VRAM: {stats['peak_allocated_gb']:.2f}GB")

# Debug logging
debug_logger = DebugLogger(log_dir="debug_logs")
# Logs are automatically saved on errors
```

## Benchmarking

```python
from utils import PerformanceBenchmark

benchmark = PerformanceBenchmark(runtime)

# Benchmark single config
result = benchmark.benchmark_config(
    prompt="Test prompt",
    num_runs=5,
    num_steps=30,
    height=512,
    width=512
)

# Compare multiple configs
configs = [
    {"num_steps": 20, "height": 512, "width": 512},
    {"num_steps": 30, "height": 512, "width": 512},
    {"num_steps": 50, "height": 512, "width": 512},
]

results = benchmark.compare_configs(
    prompt="Test prompt",
    configs=configs,
    num_runs=3
)
```

## Error Handling

The system automatically handles errors with retry logic:

```python
# Automatic retry with progressive degradation
result = runtime.generate_sync(
    prompt="Complex scene",
    num_steps=50,
    height=768,  # Might cause OOM
    width=768,
    batch_size=2
)

# Check if retries occurred
print(f"Retries: {result.retry_count}")
print(f"Status: {result.status}")
```

## Troubleshooting

### Out of Memory Errors

1. Use preset config:
```python
from config import PresetConfigs
config = PresetConfigs.low_vram()
```

2. Reduce parameters:
```python
result = runtime.generate_sync(
    prompt="...",
    num_steps=20,  # Reduce from 50
    height=512,    # Reduce from 768
    width=512,
    batch_size=1   # Reduce from 4
)
```

3. Enable all optimizations in config:
```python
config.enable_attention_slicing = True
config.enable_vae_tiling = True
config.enable_xformers = True
```

### Slow Generation

Check if CPU offloading is causing overhead:
```python
# Disable for faster inference (if you have enough VRAM)
config.enable_cpu_offload = False
```

### Check System Status

```python
stats = runtime.get_stats()
print(f"Queue size: {stats['queue']['queue_size']}")
print(f"VRAM usage: {stats['memory']['vram']['utilization_percent']:.1f}%")
print(f"Success rate: {stats['queue']['success_rate']*100:.1f}%")
```

## Advanced Usage

### Custom Execution Config

```python
from job_queue_manager import JobRequest, JobPriority
from execution_state_machine import ExecutionConfig

# Custom execution config
exec_config = ExecutionConfig(
    max_retries=5,
    retry_delay_seconds=3.0,
    timeout_seconds=600,
    enable_progressive_degradation=True,
    reduce_batch_size_on_oom=True,
    enable_tiling_on_oom=True
)

# Create job with custom config
job = JobRequest(
    prompt="...",
    execution_config=exec_config,
    priority=JobPriority.URGENT
)

job_id = runtime.submit_job(job)
```

### Direct Engine Access

```python
from diffusion_engine import DiffusionInferenceEngine
from diffusers import StableDiffusionPipeline

engine = DiffusionInferenceEngine(
    model_id="runwayml/stable-diffusion-v1-5"
)
engine.initialize(StableDiffusionPipeline)

# Direct generation (without queue)
result = engine.generate(
    prompt="...",
    num_inference_steps=50,
    height=512,
    width=512
)

# Access metrics
status = engine.state_machine.get_status()
print(status['context']['metrics'])

engine.cleanup()
```

## Next Steps

- See `example_usage.py` for complete examples
- Read `README.md` for detailed documentation
- Check `config.py` for all configuration options
- Use `utils.py` for monitoring and benchmarking
