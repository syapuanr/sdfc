# Struktur Folder dan File - Diffusion Runtime

## 📁 Struktur Lengkap

```
diffusion_runtime/
│
├── 📄 README.md                      # Main README (project overview)
├── 📄 LICENSE                        # MIT License
├── 📄 setup.py                       # Package installation script
├── 📄 requirements.txt               # Production dependencies
├── 📄 requirements-dev.txt           # Development dependencies
├── 📄 .gitignore                     # Git ignore rules
├── 📄 __init__.py                    # Package init (main imports)
│
├── 📂 src/                           # Source code
│   │
│   ├── 📂 core/                      # Core engine modules
│   │   ├── 📄 __init__.py            # Core package init
│   │   ├── 📄 memory_manager.py      # VRAM monitoring & dynamic offloading
│   │   ├── 📄 model_loader.py        # Phase-based model loading
│   │   ├── 📄 execution_state_machine.py  # State management & retry logic
│   │   ├── 📄 job_queue_manager.py   # Job scheduling & queue system
│   │   └── 📄 diffusion_engine.py    # Main inference engine
│   │
│   ├── 📂 config/                    # Configuration
│   │   ├── 📄 __init__.py            # Config package init
│   │   ├── 📄 config.py              # RuntimeConfig & detection
│   │   └── 📄 presets.py             # Preset configurations
│   │
│   └── 📂 utils/                     # Utilities
│       ├── 📄 __init__.py            # Utils package init
│       ├── 📄 monitoring.py          # Memory monitor & benchmarking
│       ├── 📄 benchmarking.py        # Performance benchmarking
│       └── 📄 logging_utils.py       # Logging utilities
│
├── 📂 examples/                      # Example scripts
│   ├── 📄 README.md                  # Examples documentation
│   ├── 📄 example_usage.py           # Complete runtime examples
│   ├── 📄 basic_generation.py        # Basic single image generation
│   ├── 📄 batch_processing.py        # Batch job processing
│   ├── 📄 with_job_queue.py          # Job queue usage
│   └── 📄 custom_config.py           # Custom configuration
│
├── 📂 tests/                         # Unit tests
│   ├── 📄 __init__.py                # Tests package init
│   ├── 📄 test_memory_manager.py     # Memory manager tests
│   ├── 📄 test_model_loader.py       # Model loader tests
│   ├── 📄 test_state_machine.py      # State machine tests
│   └── 📄 test_job_queue.py          # Job queue tests
│
├── 📂 docs/                          # Documentation
│   ├── 📄 README.md                  # Full documentation
│   ├── 📄 QUICKSTART.md              # Quick start guide
│   ├── 📄 SYSTEM_SUMMARY.md          # System overview
│   ├── 📄 API_REFERENCE.md           # API reference
│   ├── 📄 ARCHITECTURE.md            # Architecture details
│   └── 📄 TROUBLESHOOTING.md         # Troubleshooting guide
│
├── 📂 outputs/                       # Generated images (gitignored)
│   └── 📄 .gitkeep                   # Keep folder in git
│
└── 📂 logs/                          # Log files (gitignored)
    └── 📄 .gitkeep                   # Keep folder in git
```

---

## 📋 File Descriptions

### Root Level

| File | Description | Lines | Purpose |
|------|-------------|-------|---------|
| `README.md` | Main project README | ~80 | Project overview, quick start |
| `setup.py` | Package setup script | ~70 | Installation configuration |
| `requirements.txt` | Dependencies | ~15 | Production packages |
| `.gitignore` | Git ignore rules | ~60 | Files to exclude from git |
| `__init__.py` | Package entry point | ~100 | Main imports & exports |

### src/core/ - Core Engine (~3,200 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `memory_manager.py` | ~500 | VRAM monitoring, cleanup, offloading |
| `model_loader.py` | ~500 | Phase-based model loading |
| `execution_state_machine.py` | ~650 | State management, retry logic |
| `job_queue_manager.py` | ~550 | Job queue, scheduler, manager |
| `diffusion_engine.py` | ~700 | Main inference engine |
| `__init__.py` | ~80 | Core package exports |

**Total Core Code:** ~2,980 lines

### src/config/ - Configuration (~400 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | ~350 | RuntimeConfig, presets, auto-detection |
| `__init__.py` | ~20 | Config package exports |

### src/utils/ - Utilities (~450 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `monitoring.py` | ~400 | Memory monitor, benchmarking, debug logger |
| `__init__.py` | ~20 | Utils package exports |

### examples/ - Examples (~1,000 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `example_usage.py` | ~700 | Complete DiffusionRuntime with examples |
| `basic_generation.py` | ~100 | Simple generation example |
| `batch_processing.py` | ~100 | Batch processing example |
| `with_job_queue.py` | ~80 | Job queue example |

### docs/ - Documentation (~2,000 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `README.md` | ~800 | Comprehensive documentation |
| `QUICKSTART.md` | ~300 | Quick start guide |
| `SYSTEM_SUMMARY.md` | ~600 | System architecture overview |
| `API_REFERENCE.md` | ~200 | API documentation |
| `TROUBLESHOOTING.md` | ~100 | Common issues & solutions |

---

## 📊 Statistics

```
Total Files:        35+
Total Python Code:  ~4,800 lines
Total Documentation: ~2,000 lines
Total Lines:        ~7,000+ lines

Core Modules:       5 files
Configuration:      2 files
Utilities:          3 files
Examples:           5 files
Tests:              5 files
Documentation:      6 files
```

---

## 🔧 Installation & Setup

### Method 1: Development Install (Editable)
```bash
# Clone/download the folder
cd diffusion_runtime

# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install with xformers
pip install -e ".[xformers]"
```

### Method 2: Direct Install
```bash
# From folder
pip install .

# Or build wheel
python setup.py bdist_wheel
pip install dist/diffusion_runtime-1.0.0-*.whl
```

### Method 3: Just Copy Files
```bash
# Simply copy the folder and import
cp -r diffusion_runtime /path/to/your/project/
cd /path/to/your/project/
python -c "from diffusion_runtime import DiffusionRuntime"
```

---

## 📦 Package Structure untuk Import

Setelah install, struktur import:

```python
# Main runtime
from diffusion_runtime import DiffusionRuntime

# Core components
from diffusion_runtime.src.core import (
    MemoryManager,
    DiffusionModelLoader,
    ExecutionStateMachine,
    JobManager
)

# Configuration
from diffusion_runtime.src.config import (
    RuntimeConfig,
    PresetConfigs,
    detect_environment
)

# Utilities
from diffusion_runtime.src.utils import (
    MemoryMonitor,
    PerformanceBenchmark
)
```

---

## 🚀 Usage Patterns

### Pattern 1: Simple Import
```python
from diffusion_runtime import DiffusionRuntime
from diffusers import StableDiffusionPipeline

runtime = DiffusionRuntime("runwayml/stable-diffusion-v1-5")
runtime.start(StableDiffusionPipeline)
# ... use runtime
```

### Pattern 2: Custom Configuration
```python
from diffusion_runtime.src.config import RuntimeConfig, PresetConfigs
from diffusion_runtime import DiffusionRuntime

# Use preset
config = PresetConfigs.low_vram()

# Or custom
config = RuntimeConfig(
    model_id="stabilityai/stable-diffusion-2-1",
    max_retries=5
)

runtime = DiffusionRuntime(
    model_id=config.model_id,
    # ... use config settings
)
```

### Pattern 3: Direct Core Access
```python
from diffusion_runtime.src.core import (
    DiffusionInferenceEngine,
    MemoryManager
)

memory_manager = MemoryManager()
engine = DiffusionInferenceEngine(
    model_id="runwayml/stable-diffusion-v1-5",
    memory_manager=memory_manager
)
```

---

## 📂 Recommended Workflow

### 1. Development
```bash
diffusion_runtime/
├── Start here: examples/basic_generation.py
├── Check: docs/QUICKSTART.md
└── Modify: src/core/*.py
```

### 2. Testing
```bash
# Run examples
python examples/basic_generation.py

# Run tests
pytest tests/

# Check memory
python -c "from diffusion_runtime.src.utils import MemoryMonitor; ..."
```

### 3. Deployment
```bash
# Install package
pip install .

# Use in production
from diffusion_runtime import DiffusionRuntime
# ... production code
```

---

## 🎯 Key Files to Start With

### For Users:
1. **`README.md`** - Start here
2. **`docs/QUICKSTART.md`** - Quick tutorial
3. **`examples/basic_generation.py`** - Simple example
4. **`examples/example_usage.py`** - Complete examples

### For Developers:
1. **`src/core/__init__.py`** - See all available classes
2. **`src/core/diffusion_engine.py`** - Main engine logic
3. **`src/core/memory_manager.py`** - Memory management
4. **`docs/SYSTEM_SUMMARY.md`** - Architecture overview

### For Configuration:
1. **`src/config/config.py`** - All config options
2. **`examples/custom_config.py`** - Config examples

---

## 💡 Tips

### Folder Navigation
```bash
# Core modules (main logic)
cd src/core/

# Examples (how to use)
cd examples/

# Documentation (learn more)
cd docs/

# Configuration (customize)
cd src/config/
```

### File Finding
```bash
# Find all Python files
find . -name "*.py"

# Find documentation
find . -name "*.md"

# Count lines of code
find src/ -name "*.py" | xargs wc -l
```

### Quick Access
```bash
# Read main README
cat README.md

# Read quick start
cat docs/QUICKSTART.md

# List examples
ls examples/

# Check imports
cat __init__.py
```

---

## 📝 Notes

- **All Python files** have proper docstrings
- **All modules** have `__init__.py` for proper packaging
- **Documentation** is comprehensive and beginner-friendly
- **Examples** are self-contained and runnable
- **Tests** are structured for pytest
- **Git-friendly** with proper .gitignore

---

**Total Package Size:** ~150KB (source code only, excluding models)  
**Installation Size:** ~200KB (with compiled .pyc files)  
**Model Downloads:** Handled by diffusers library (separate)
