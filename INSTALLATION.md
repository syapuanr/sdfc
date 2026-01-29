# 🚀 Installation & Setup Guide

## Cara Install dan Gunakan Diffusion Runtime

---

## 📦 Method 1: Install sebagai Package (Recommended)

### Step 1: Ekstrak Folder
```bash
# Ekstrak folder diffusion_runtime ke lokasi project Anda
cd /path/to/your/workspace
# folder diffusion_runtime sudah ada di sini
```

### Step 2: Install Package
```bash
cd diffusion_runtime

# Install dalam mode editable (development)
pip install -e .

# Atau install dengan dependencies lengkap
pip install -e ".[dev,xformers]"
```

### Step 3: Gunakan!
```python
# Import langsung dari mana saja
from diffusion_runtime import DiffusionRuntime
from diffusers import StableDiffusionPipeline

runtime = DiffusionRuntime("runwayml/stable-diffusion-v1-5")
runtime.start(StableDiffusionPipeline)

result = runtime.generate_sync(prompt="A beautiful landscape")
result.result.images[0].save("output.png")

runtime.stop()
```

---

## 📋 Method 2: Copy Files (Simple)

Jika tidak ingin install sebagai package:

### Step 1: Copy Folder src/
```bash
# Copy folder src ke project Anda
cp -r diffusion_runtime/src /path/to/your/project/
```

### Step 2: Install Dependencies
```bash
pip install -r diffusion_runtime/requirements.txt
```

### Step 3: Import dari src/
```python
import sys
sys.path.append('/path/to/your/project')

from src.core import DiffusionInferenceEngine, MemoryManager
from src.config import RuntimeConfig

# Gunakan langsung
engine = DiffusionInferenceEngine("runwayml/stable-diffusion-v1-5")
```

---

## 🔧 Method 3: Google Colab

### Step 1: Upload Folder
```python
# Di Colab, upload folder diffusion_runtime
from google.colab import files
import zipfile

# Upload diffusion_runtime.zip
uploaded = files.upload()

# Extract
!unzip diffusion_runtime.zip
```

### Step 2: Install
```python
# Install dependencies
!pip install -q torch diffusers transformers accelerate xformers

# Install package
!cd diffusion_runtime && pip install -e .
```

### Step 3: Gunakan
```python
from diffusion_runtime import DiffusionRuntime
from diffusers import StableDiffusionPipeline

runtime = DiffusionRuntime("runwayml/stable-diffusion-v1-5")
runtime.start(StableDiffusionPipeline)

result = runtime.generate_sync(
    prompt="A serene mountain landscape at sunset",
    num_steps=30
)

from google.colab.patches import cv2_imshow
import numpy as np
cv2_imshow(np.array(result.result.images[0]))

runtime.stop()
```

---

## 📁 Struktur Setelah Install

### Method 1 (Package Install):
```
your_workspace/
├── diffusion_runtime/          # Source folder
│   ├── src/
│   ├── examples/
│   └── ...
│
└── your_script.py              # Your code
    from diffusion_runtime import DiffusionRuntime  ✓
```

### Method 2 (Copy Files):
```
your_project/
├── src/                        # Copied source
│   ├── core/
│   ├── config/
│   └── utils/
│
└── your_script.py              # Your code
    from src.core import DiffusionInferenceEngine  ✓
```

---

## 🎯 Quick Start Examples

### Example 1: Basic Generation
```python
from diffusion_runtime import DiffusionRuntime
from diffusers import StableDiffusionPipeline

# Create runtime
runtime = DiffusionRuntime(
    model_id="runwayml/stable-diffusion-v1-5",
    device="cuda"
)

# Start
runtime.start(StableDiffusionPipeline)

# Generate
result = runtime.generate_sync(
    prompt="A cat sitting on a table, photorealistic",
    negative_prompt="blurry, low quality",
    num_steps=30,
    height=512,
    width=512
)

# Save
result.result.images[0].save("cat.png")

# Cleanup
runtime.stop()
```

### Example 2: Batch Processing
```python
from diffusion_runtime import DiffusionRuntime
from diffusers import StableDiffusionPipeline

runtime = DiffusionRuntime()
runtime.start(StableDiffusionPipeline)

# Submit multiple jobs
prompts = [
    "A sunset over mountains",
    "A city at night",
    "A forest in autumn"
]

job_ids = []
for prompt in prompts:
    job_id = runtime.submit_job(
        prompt=prompt,
        num_steps=25
    )
    job_ids.append(job_id)
    print(f"Submitted: {prompt}")

# Wait for completion
import time
while True:
    stats = runtime.get_stats()
    if stats['queue']['queue_size'] == 0:
        break
    print(f"Queue: {stats['queue']['queue_size']} remaining")
    time.sleep(2)

# Get results
for i, job_id in enumerate(job_ids):
    result = runtime.get_job_result(job_id)
    if result.result:
        result.result.images[0].save(f"image_{i}.png")
        print(f"Saved: image_{i}.png")

runtime.stop()
```

### Example 3: Auto-Config
```python
from diffusion_runtime import DiffusionRuntime
from diffusion_runtime.src.config import detect_environment
from diffusers import StableDiffusionPipeline

# Auto-detect best config for your system
config = detect_environment()
print(f"Detected config: {config.model_id}")
print(f"VRAM optimizations: {config.enable_cpu_offload}")

# Use auto-config
runtime = DiffusionRuntime(
    model_id=config.model_id,
    device=str(config.device)
)

runtime.start(StableDiffusionPipeline)

result = runtime.generate_sync(
    prompt="Your prompt here",
    num_steps=30
)

result.result.images[0].save("output.png")
runtime.stop()
```

---

## ⚙️ Configuration Options

### Low VRAM (4-6GB)
```python
from diffusion_runtime.src.config import PresetConfigs

config = PresetConfigs.low_vram()

runtime = DiffusionRuntime(
    model_id=config.model_id,
    device=str(config.device)
)
```

### High Performance (12GB+)
```python
config = PresetConfigs.high_vram()
# Disables offloading for speed
```

### Custom Config
```python
from diffusion_runtime.src.config import RuntimeConfig

config = RuntimeConfig(
    model_id="stabilityai/stable-diffusion-2-1",
    enable_cpu_offload=True,
    max_retries=5,
    enable_progressive_degradation=True,
    safe_threshold=0.70
)
```

---

## 🐛 Troubleshooting

### Import Error
```bash
# Make sure package is installed
pip install -e diffusion_runtime/

# Or check if in Python path
python -c "import sys; print(sys.path)"
```

### CUDA Out of Memory
```python
# Use low VRAM preset
from diffusion_runtime.src.config import PresetConfigs
config = PresetConfigs.low_vram()

# Reduce parameters
result = runtime.generate_sync(
    num_steps=20,    # Lower steps
    height=512,      # Lower resolution
    batch_size=1     # Single image
)
```

### Slow Generation
```python
# Disable CPU offloading if you have enough VRAM
config = RuntimeConfig(
    enable_cpu_offload=False,  # Faster but needs more VRAM
    enable_xformers=True        # Use xformers for speed
)
```

---

## 📚 Next Steps

1. **Read Documentation**:
   - `docs/QUICKSTART.md` - Quick tutorial
   - `docs/README.md` - Full documentation
   - `docs/SYSTEM_SUMMARY.md` - Architecture

2. **Try Examples**:
   - `examples/basic_generation.py`
   - `examples/batch_processing.py`
   - `examples/custom_config.py`

3. **Explore Code**:
   - `src/core/` - Core modules
   - `src/config/` - Configuration
   - `src/utils/` - Utilities

---

## 💡 Tips

### Tip 1: Check Memory
```python
from diffusion_runtime.src.utils import MemoryMonitor

monitor = MemoryMonitor()
monitor.start()

# Run your generation...

stats = monitor.stop()
print(f"Peak VRAM: {stats['peak_allocated_gb']:.2f}GB")
```

### Tip 2: Benchmark Performance
```python
from diffusion_runtime.src.utils import PerformanceBenchmark

benchmark = PerformanceBenchmark(runtime)

results = benchmark.benchmark_config(
    prompt="Test prompt",
    num_runs=5,
    num_steps=30
)

print(f"Average time: {results.avg_time:.2f}s")
```

### Tip 3: Monitor Queue
```python
stats = runtime.get_stats()
print(f"Queue size: {stats['queue']['queue_size']}")
print(f"Success rate: {stats['queue']['success_rate']*100:.1f}%")
print(f"VRAM usage: {stats['memory']['vram']['utilization_percent']:.1f}%")
```

---

## 🎉 You're Ready!

Your diffusion runtime is now installed and ready to use. Happy generating! 🚀

For questions:
- Check `docs/` folder
- Read `FOLDER_STRUCTURE.md`
- Try `examples/` scripts
