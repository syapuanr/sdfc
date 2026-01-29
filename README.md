# SDFC (Stable Diffusion For Colab)

> **Fault-Tolerant Diffusion Inference System** optimized for limited VRAM environments.

## 🎯 Overview

**SDFC** (Stable Diffusion For Colab) adalah sistem inferensi difusi yang *production-ready*, dioptimalkan khusus untuk lingkungan dengan VRAM terbatas seperti Google Colab, Kaggle, atau Local GPU (4GB-8GB VRAM).

### ✨ Key Features

✅ **Phase-Based Model Loading** (CPU Offload otomatis saat idle)  
✅ **Automatic OOM Recovery** (Sistem anti-crash yang pintar)  
✅ **Job Queue System** (Manajemen antrean prioritas)  
✅ **Real-time Monitoring** (Pemantauan penggunaan VRAM live)  
✅ **Production Ready** (Arsitektur modular dan mudah dikembangkan)
## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone [https://github.com/syapuanr/sdfc.git](https://github.com/syapuanr/sdfc.git)
cd sdfc

# Install dependencies
pip install -e .
from diffusion_runtime.src.core.diffusion_engine import DiffusionInferenceEngine
from diffusers import StableDiffusionPipeline

# Inisialisasi Engine (Mode Hemat VRAM aktif)
engine = DiffusionInferenceEngine(
    model_id="runwayml/stable-diffusion-v1-5",
    enable_cpu_offload=True
)

# Load Pipeline
print("⏳ Loading model...")
engine.initialize(StableDiffusionPipeline)

# Generate Image
print("🎨 Generating image...")
result = engine.generate(
    prompt="A futuristic city with neon lights, cyberpunk style, 8k resolution",
    num_inference_steps=30,
    guidance_scale=7.5
)

# Save Output
if result.images:
    output_path = "output_sdfc.png"
    result.images[0].save(output_path)
    print(f"✅ Gambar berhasil disimpan di: {output_path}")
---

### ✂️ BAGIAN 3: Footer & Struktur (Paste paling bawah)

```markdown
## 🔗 Repository

**GitHub:** [github.com/syapuanr/sdfc](https://github.com/syapuanr/sdfc)

## 📖 Documentation

- [Full Documentation](docs/README.md)
- [Quick Start Guide](docs/QUICKSTART.md)  
- [System Summary](docs/SYSTEM_SUMMARY.md)

## 📁 Project Structure
sdfc/
├── src/
│   ├── core/          # Core Engine
│   ├── config/        # Configuration
│   └── utils/         # Monitoring
├── examples/          # Contoh script
├── docs/              # Dokumentasi
└── outputs/           # Hasil gambar
Lihat [docs/SYSTEM_SUMMARY.md](docs/SYSTEM_SUMMARY.md) untuk detail arsitektur.

## 📄 License

MIT License
