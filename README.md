# Diffusion Runtime

> **Fault-Tolerant Diffusion Inference System** untuk lingkungan dengan memori GPU terbatas

## 🎯 Overview

Diffusion Runtime adalah sistem inferensi difusi yang production-ready, dioptimalkan untuk lingkungan dengan VRAM terbatas seperti Google Colab.

### Key Features

✅ Phase-Based Model Loading  
✅ Automatic OOM Recovery  
✅ Job Queue System  
✅ Real-time Monitoring  
✅ Production Ready  

## 🚀 Quick Start

```bash
pip install -e .
```

```python
from diffusion_runtime import DiffusionRuntime
from diffusers import StableDiffusionPipeline

runtime = DiffusionRuntime("runwayml/stable-diffusion-v1-5")
runtime.start(StableDiffusionPipeline)

result = runtime.generate_sync(prompt="A beautiful sunset")
result.result.images[0].save("output.png")

runtime.stop()
```

## 🔗 Repository

**GitHub:** [github.com/syapuanr/diffusion-runtime](https://github.com/syapuanr/diffusion-runtime)

## 📖 Documentation

- [Full Documentation](docs/README.md)
- [Quick Start Guide](docs/QUICKSTART.md)  
- [System Summary](docs/SYSTEM_SUMMARY.md)

## 📁 Structure

```
diffusion_runtime/
├── src/core/          # Core modules
├── src/config/        # Configuration
├── src/utils/         # Utilities
├── examples/          # Examples
├── tests/             # Tests
└── docs/              # Documentation
```

See [docs/SYSTEM_SUMMARY.md](docs/SYSTEM_SUMMARY.md) for details.

## 📄 License

MIT License
