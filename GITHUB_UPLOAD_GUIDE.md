# 🚀 GitHub Upload Guide

## Upload Project ke GitHub

Panduan lengkap untuk upload **diffusion-runtime** ke GitHub repository Anda.

---

## 📋 Prerequisites

1. ✅ GitHub account: **syapuanr**
2. ✅ Git installed di komputer
3. ✅ Folder `diffusion_runtime` sudah ready

---

## 🎯 Method 1: Via GitHub Web Interface (Termudah)

### Step 1: Buat Repository Baru

1. Buka [github.com](https://github.com)
2. Login dengan username: **syapuanr**
3. Click **"New repository"** (tombol hijau)
4. Isi form:
   - **Repository name:** `diffusion-runtime`
   - **Description:** `Fault-Tolerant Diffusion Inference Runtime for Memory-Constrained Environments`
   - **Visibility:** Public (atau Private jika mau)
   - **❌ JANGAN** centang "Initialize with README" (kita sudah punya)
5. Click **"Create repository"**

### Step 2: Upload Files

**Option A: Drag & Drop (Termudah)**
1. Di halaman repository baru, click **"uploading an existing file"**
2. Drag & drop seluruh isi folder `diffusion_runtime/`
3. Atau: Click **"choose your files"** dan pilih semua file
4. Isi commit message: `Initial commit - Complete diffusion runtime system`
5. Click **"Commit changes"**

**Option B: Upload ZIP**
1. Compress folder `diffusion_runtime` menjadi ZIP
2. Extract di local setelah clone

---

## 🎯 Method 2: Via Git Command Line (Recommended)

### Step 1: Buat Repository di GitHub
(Sama seperti Method 1 - Step 1)

### Step 2: Initialize Local Git

```bash
cd diffusion_runtime

# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: Complete diffusion runtime system

- Phase-based memory management
- Automatic OOM recovery
- Job queue system
- Comprehensive monitoring
- Production-ready features"
```

### Step 3: Connect ke GitHub

```bash
# Add remote (ganti dengan URL repo Anda)
git remote add origin https://github.com/syapuanr/diffusion-runtime.git

# Rename branch to main (jika masih master)
git branch -M main

# Push to GitHub
git push -u origin main
```

### Step 4: Enter Credentials

Saat diminta credentials:
- **Username:** `syapuanr`
- **Password:** Personal Access Token (bukan password GitHub)

**Cara buat Personal Access Token:**
1. GitHub → Settings → Developer settings
2. Personal access tokens → Tokens (classic)
3. Generate new token
4. Pilih scopes: `repo` (full control)
5. Copy token (SIMPAN! tidak akan muncul lagi)

---

## 🎯 Method 3: Via GitHub Desktop (GUI)

### Step 1: Install GitHub Desktop
Download dari [desktop.github.com](https://desktop.github.com)

### Step 2: Login
Login dengan username **syapuanr**

### Step 3: Add Repository
1. File → Add local repository
2. Pilih folder `diffusion_runtime`
3. Jika belum git repo, click "Create a repository"
4. Isi nama: `diffusion-runtime`

### Step 4: Publish
1. Click **"Publish repository"** di toolbar
2. Pilih visibility (Public/Private)
3. Click **"Publish repository"**

✅ Done!

---

## 📝 Setelah Upload

### Verify Upload Berhasil

Buka: `https://github.com/syapuanr/diffusion-runtime`

Harus melihat:
```
diffusion-runtime/
├── src/
├── examples/
├── docs/
├── tests/
├── setup.py
├── requirements.txt
└── README.md
```

### Set Repository Description

Di halaman repo:
1. Click ⚙️ (Settings)
2. Isi **Description:** 
   ```
   Fault-Tolerant Diffusion Inference Runtime for Memory-Constrained Environments | Phase-based Loading | OOM Recovery | Job Queue
   ```
3. Isi **Website:** (optional)
4. Add **Topics/Tags:**
   - `diffusion-models`
   - `stable-diffusion`
   - `pytorch`
   - `gpu-memory`
   - `google-colab`
   - `inference-engine`

---

## 🎨 Optional: Add Badges ke README

Edit `README.md` dan tambahkan di bagian atas:

```markdown
# Diffusion Runtime

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/syapuanr/diffusion-runtime.svg)](https://github.com/syapuanr/diffusion-runtime/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/syapuanr/diffusion-runtime.svg)](https://github.com/syapuanr/diffusion-runtime/issues)
```

---

## 📦 Update Setup.py dengan GitHub URL

Edit `setup.py`, ganti URL:

```python
setup(
    name="diffusion-runtime",
    version="1.0.0",
    author="syapuanr",
    url="https://github.com/syapuanr/diffusion-runtime",
    # ...
)
```

Commit perubahan:
```bash
git add setup.py README.md
git commit -m "Update repository URLs and add badges"
git push
```

---

## 🎯 Clone Repository (Test)

Test apakah orang lain bisa clone:

```bash
# Clone
git clone https://github.com/syapuanr/diffusion-runtime.git

# Install
cd diffusion-runtime
pip install -r requirements.txt
pip install -e .

# Test
python -c "from diffusion_runtime import DiffusionRuntime; print('✓ OK')"
```

---

## 📚 Add Installation Instructions ke README

Tambahkan section di README.md:

```markdown
## 🚀 Installation

### From GitHub

\`\`\`bash
# Clone repository
git clone https://github.com/syapuanr/diffusion-runtime.git
cd diffusion-runtime

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
\`\`\`

### From PyPI (future)

\`\`\`bash
pip install diffusion-runtime
\`\`\`
```

---

## 🔄 Future Updates

Untuk update repository:

```bash
# Make changes to code
# ...

# Stage changes
git add .

# Commit
git commit -m "Description of changes"

# Push
git push
```

---

## 🎉 Repository Setup Complete!

Your repository akan tersedia di:
**https://github.com/syapuanr/diffusion-runtime**

People can:
- ⭐ Star your repo
- 🍴 Fork your repo  
- 📥 Clone your repo
- 🐛 Report issues
- 🔧 Submit pull requests

---

## 📞 Quick Links

- **Repository:** https://github.com/syapuanr/diffusion-runtime
- **Clone URL:** `git clone https://github.com/syapuanr/diffusion-runtime.git`
- **Issues:** https://github.com/syapuanr/diffusion-runtime/issues
- **Releases:** https://github.com/syapuanr/diffusion-runtime/releases

---

## 💡 Pro Tips

1. **Create Releases:**
   - Tag your versions: `git tag v1.0.0`
   - Push tags: `git push --tags`
   - Create release on GitHub

2. **Enable GitHub Pages:**
   - Settings → Pages
   - Deploy docs dari `/docs` folder

3. **Add CI/CD:**
   - Create `.github/workflows/test.yml`
   - Auto-run tests on push

4. **Add Contributing Guide:**
   - Create `CONTRIBUTING.md`
   - Guidelines for contributors

Happy sharing! 🚀
