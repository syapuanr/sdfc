# Fault-Tolerant Diffusion Inference Runtime

Runtime inferensi difusi yang toleran terhadap kesalahan, dioptimalkan untuk lingkungan dengan memori GPU terbatas seperti Google Colab.

## 🎯 Fitur Utama

### 1. **Memory Management yang Adaptif**
- ✅ Monitoring VRAM real-time dengan threshold detection
- ✅ Dynamic CPU ↔ GPU offloading
- ✅ Aggressive garbage collection dan cache cleanup
- ✅ Phase-based model loading (Text Encoder → UNet → VAE tidak bersamaan)
- ✅ Memory estimation sebelum operasi

### 2. **Fault Tolerance & Recovery**
- ✅ OOM (Out of Memory) error detection dan recovery
- ✅ Progressive degradation pada retry (batch size → tiling → resolution)
- ✅ Automatic retry dengan exponential backoff
- ✅ CUDA error recovery
- ✅ Comprehensive error classification

### 3. **Memory Optimization Techniques**
- ✅ Attention slicing untuk reduced memory footprint
- ✅ VAE tiling untuk decode image besar
- ✅ XFormers/Flash Attention support
- ✅ Gradient checkpointing compatible
- ✅ FP16 precision untuk reduced memory

### 4. **Job Queue & Scheduling**
- ✅ Priority-based job queue
- ✅ Multi-threaded job execution
- ✅ Job status tracking dan monitoring
- ✅ Callback support (on_progress, on_complete, on_error)
- ✅ Comprehensive job metrics

### 5. **Execution State Machine**
- ✅ Explicit state transitions untuk debugging
- ✅ Progress tracking per step
- ✅ Health monitoring during inference
- ✅ Detailed metrics collection (timing, memory, retries)
- ✅ Intermediate result checkpointing

---

## 📐 Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                           │
│          (submit_job, get_result, get_status)                │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Job Queue Manager                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Priority Queue  │  Active Jobs  │  Completed Jobs   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Execution State Machine                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ IDLE → LOADING → ENCODING → DIFFUSION → DECODING    │  │
│  │   ↓                                             ↓     │  │
│  │ RETRY_PENDING ← FAILED                    COMPLETED  │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│            Diffusion Inference Engine                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Phase 1: Text Encoding                              │  │
│  │  Phase 2: UNet Diffusion (with progress tracking)    │  │
│  │  Phase 3: VAE Decoding (with tiling fallback)       │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Model Loader Layer                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  prepare_text_encoder() - Load text encoder ke GPU  │  │
│  │  prepare_unet() - Load UNet, offload text encoder   │  │
│  │  prepare_vae() - Load VAE, offload UNet             │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│            Memory Management Layer                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  VRAMMonitor: Real-time monitoring & thresholds      │  │
│  │  MemoryManager: Cleanup, offloading, optimization    │  │
│  │  MemoryOptimizer: Attention/VAE slicing & tiling     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    PyTorch & CUDA                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Execution Flow

### Normal Execution Path
```
1. User submits job
   ↓
2. Job enqueued dengan priority
   ↓
3. Worker dequeues job
   ↓
4. State Machine: IDLE → INITIALIZING
   ↓
5. Phase 1: ENCODING_PROMPT
   - Load text encoder ke GPU
   - Offload UNet & VAE
   - Encode prompt & negative prompt
   - Move embeddings ke GPU
   ↓
6. Phase 2: DIFFUSION_RUNNING
   - Offload text encoder
   - Load UNet ke GPU
   - Run diffusion loop (with progress tracking)
   - Monitor VRAM per step
   ↓
7. Phase 3: DECODING
   - Offload UNet
   - Load VAE ke GPU
   - Decode latents ke images
   - (Fallback to tiled decode jika OOM)
   ↓
8. State Machine: COMPLETED
   ↓
9. Return result dengan images & metrics
```

### Error Recovery Path
```
OOM Error Detected
   ↓
State Machine: FAILED → RETRY_PENDING
   ↓
Retry Attempt 1: Reduce batch size
   ↓
Still OOM?
   ↓
Retry Attempt 2: Enable aggressive tiling
   ↓
Still OOM?
   ↓
Retry Attempt 3: Reduce resolution
   ↓
Still failing? → FAILED (permanent)
```

---

## 📦 Modul-Modul

### 1. `memory_manager.py`
**Menangani semua aspek memory management**

#### VRAMMonitor
- Real-time VRAM monitoring
- Threshold detection (safe, warning, critical)
- Memory availability checking
- Logging memory statistics

#### MemoryManager
- Model registration dan tracking
- CPU ↔ GPU offloading
- Aggressive cleanup (GC + CUDA cache)
- Memory space reservation

#### MemoryOptimizer
- Attention slicing configuration
- VAE slicing/tiling
- XFormers integration
- Optimal tile size calculation

**Key Methods:**
```python
monitor = VRAMMonitor(device="cuda:0")
stats = monitor.get_memory_stats()
can_fit = monitor.can_fit(required_gb=2.5)

manager = MemoryManager(device="cuda:0")
manager.offload_to_cpu(model, "unet")
manager.load_to_gpu(model, "text_encoder")
manager.aggressive_cleanup()
```

---

### 2. `model_loader.py`
**Phase-based model loading dengan lazy loading support**

#### DiffusionModelLoader
- Load pipeline dari diffusers
- Extract individual components
- Apply memory optimizations
- Phase-based GPU loading

**Phases:**
1. `prepare_text_encoder()` - Load text encoder, offload others
2. `prepare_unet()` - Load UNet, offload text encoder & VAE
3. `prepare_vae()` - Load VAE, offload UNet & text encoder

**Key Methods:**
```python
config = ModelConfig(
    model_id="runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    enable_cpu_offload=True
)

loader = DiffusionModelLoader(config, memory_manager)
loader.load_pipeline_from_diffusers(StableDiffusionPipeline)

# Phase-based loading
text_encoder = loader.prepare_text_encoder()
unet = loader.prepare_unet()
vae = loader.prepare_vae()
```

---

### 3. `execution_state_machine.py`
**Mengelola execution lifecycle dengan retry logic**

#### ExecutionState (States)
- `IDLE` - No active execution
- `INITIALIZING` - Setting up execution
- `ENCODING_PROMPT` - Text encoding phase
- `DIFFUSION_RUNNING` - UNet diffusion phase
- `DECODING` - VAE decoding phase
- `COMPLETED` - Success
- `FAILED` - Failed permanently
- `RETRY_PENDING` - Preparing to retry

#### ExecutionStateMachine
- State transition management
- Error classification (OOM, CUDA, timeout, etc)
- Retry logic dengan progressive degradation
- Metrics collection

#### OOMRecoveryStrategy
- `reduce_batch_size()` - Cut batch in half
- `enable_more_tiling()` - Aggressive VAE tiling
- `reduce_resolution()` - Last resort
- `progressive_degradation()` - Apply strategies sequentially

**Key Methods:**
```python
state_machine = ExecutionStateMachine(memory_manager)
context = ExecutionContext(job_id="job-001", prompt="...", num_steps=50)

state_machine.start_execution(context)
# ... run phases ...
state_machine.complete_execution(result)

# Error handling
state_machine._handle_error(exception)  # Auto-retry dengan degradation
```

---

### 4. `job_queue_manager.py`
**Priority queue dan job scheduling**

#### JobRequest
- Job parameters (prompt, size, steps, etc)
- Priority level (LOW, NORMAL, HIGH, URGENT)
- Callbacks (on_progress, on_complete, on_error)
- Execution config (max retries, timeout, etc)

#### JobQueue
- Thread-safe priority queue
- Active job tracking
- Completed job history
- Statistics (total, success rate, etc)

#### JobScheduler
- Worker thread management
- Job dispatching
- Timeout handling
- Graceful shutdown

#### JobManager
- High-level API
- Submit jobs, get status, get results
- Queue statistics

**Key Methods:**
```python
job_manager = JobManager(
    executor_fn=execute_fn,
    max_queue_size=100,
    num_workers=1
)

job_manager.start()

job = JobRequest(
    prompt="A beautiful landscape",
    priority=JobPriority.HIGH,
    num_inference_steps=50
)

job_id = job_manager.submit_job(job)
status = job_manager.get_job_status(job_id)
result = job_manager.get_job_result(job_id)
```

---

### 5. `diffusion_engine.py`
**Main inference engine yang mengintegrasikan semua komponen**

#### DiffusionInferenceEngine
- Complete inference pipeline
- Phase-based execution
- Automatic retry dengan fallback
- Comprehensive metrics

**Execution Phases:**
```python
engine = DiffusionInferenceEngine(
    model_id="runwayml/stable-diffusion-v1-5",
    enable_cpu_offload=True
)

engine.initialize(StableDiffusionPipeline)

result = engine.generate(
    prompt="A serene landscape",
    negative_prompt="blurry, ugly",
    num_inference_steps=50,
    height=512,
    width=512,
    seed=42
)

# Result contains:
# - images: List[PIL.Image]
# - latents: torch.Tensor
# - metrics: timing, memory, retries
```

**Internal Flow:**
1. `_encode_prompt()` - Phase 1
2. `_run_diffusion()` - Phase 2 dengan progress tracking
3. `_decode_latents()` - Phase 3 dengan tiling fallback
4. `_decode_latents_tiled()` - Fallback untuk OOM

---

### 6. `example_usage.py`
**Complete runtime dengan job queue integration**

#### DiffusionRuntime
- Combines engine + job queue
- Synchronous dan asynchronous generation
- Batch job processing
- Statistics dan monitoring

**Examples:**
```python
runtime = DiffusionRuntime(model_id="runwayml/stable-diffusion-v1-5")
runtime.start(StableDiffusionPipeline)

# Synchronous generation
result = runtime.generate_sync(
    prompt="A cat sitting on a table",
    num_steps=30
)

# Asynchronous (via queue)
job_id = runtime.submit_job(
    prompt="A dog running in a park",
    priority=JobPriority.HIGH
)

# Check status
status = runtime.get_job_status(job_id)
result = runtime.get_job_result(job_id)

runtime.stop()
```

---

## 🚀 Quick Start

### Installation
```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install pillow numpy

# Optional: untuk memory efficient attention
pip install xformers
```

### Basic Usage
```python
import torch
from diffusers import StableDiffusionPipeline
from example_usage import DiffusionRuntime

# Create runtime
runtime = DiffusionRuntime(
    model_id="runwayml/stable-diffusion-v1-5",
    device="cuda"
)

# Start
runtime.start(StableDiffusionPipeline)

# Generate
result = runtime.generate_sync(
    prompt="A beautiful sunset over mountains",
    num_steps=30,
    height=512,
    width=512
)

# Save
result.result.images[0].save("output.png")

# Stop
runtime.stop()
```

---

## 🧪 Testing

### Test Memory Management
```python
from memory_manager import MemoryManager

manager = MemoryManager(device="cuda:0")

# Check initial state
report = manager.get_memory_report()
print(f"VRAM: {report['vram']['allocated_gb']:.2f}GB")

# Test cleanup
manager.aggressive_cleanup()

# Test offloading
model = model.to("cpu")  # Manual offload
manager.offload_to_cpu(model, "model_name")  # Managed offload
```

### Test State Machine
```python
from execution_state_machine import ExecutionStateMachine, ExecutionContext

state_machine = ExecutionStateMachine(memory_manager)
context = ExecutionContext(job_id="test", prompt="test", num_steps=10)

state_machine.start_execution(context)
# Simulate error
state_machine._handle_error(MemoryError("OOM"))
# Will auto-retry dengan adjusted parameters
```

### Test Job Queue
```python
from job_queue_manager import JobManager, JobRequest

def executor(job):
    print(f"Executing {job.job_id}")
    return JobResult(job_id=job.job_id, status=JobStatus.COMPLETED)

manager = JobManager(executor_fn=executor)
manager.start()

job = JobRequest(prompt="test")
job_id = manager.submit_job(job)

# Wait and check
result = manager.get_job_result(job_id)
print(result.status)

manager.stop()
```

---

## 💡 Best Practices

### 1. Memory Management
```python
# Always use context manager atau try-finally
runtime = DiffusionRuntime()
try:
    runtime.start(pipeline_class)
    # ... operations ...
finally:
    runtime.stop()  # Ensures cleanup

# Monitor memory regularly
stats = runtime.get_stats()
if stats['memory']['vram']['utilization_percent'] > 90:
    print("High memory usage!")
```

### 2. Error Handling
```python
# Set appropriate retry limits
job = JobRequest(
    prompt="...",
    execution_config=ExecutionConfig(
        max_retries=3,
        enable_progressive_degradation=True,
        reduce_batch_size_on_oom=True
    )
)

# Handle failures gracefully
result = runtime.get_job_result(job_id)
if result.status == JobStatus.FAILED:
    print(f"Failed after {result.retry_count} retries")
    print(f"Error: {result.error_message}")
```

### 3. Optimization Tips
```python
# For low VRAM (< 8GB):
config = ModelConfig(
    model_id="stabilityai/stable-diffusion-2-1-base",  # Smaller than XL
    enable_attention_slicing=True,
    enable_vae_tiling=True,
    enable_xformers=True,
    torch_dtype=torch.float16  # Half precision
)

# Start with conservative parameters
result = runtime.generate_sync(
    prompt="...",
    num_steps=25,  # Fewer steps
    height=512,    # Standard resolution
    width=512,
    batch_size=1   # Single image
)
```

### 4. Batch Processing
```python
# Submit multiple jobs dengan priorities
urgent_job = runtime.submit_job(
    prompt="Time-sensitive content",
    priority=JobPriority.URGENT
)

# Normal jobs akan di-process setelah urgent
normal_jobs = [
    runtime.submit_job(prompt=p, priority=JobPriority.NORMAL)
    for p in prompts_list
]

# Monitor progress
while True:
    stats = runtime.get_stats()
    if stats['queue']['queue_size'] == 0:
        break
    print(f"Queue: {stats['queue']['queue_size']} remaining")
```

---

## 🐛 Troubleshooting

### OOM Errors
**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Enable all optimizations:
   ```python
   enable_attention_slicing=True
   enable_vae_tiling=True
   enable_xformers=True
   ```

2. Reduce parameters:
   ```python
   num_steps=20  # Instead of 50
   height=512    # Instead of 768
   batch_size=1  # Instead of 4
   ```

3. Use smaller models:
   - SD 1.5 (4GB VRAM)
   - SD 2.1 base (6GB VRAM)
   - Avoid SDXL (12GB+ VRAM)

### Slow Generation
**Symptom:** Generation takes too long

**Solutions:**
1. Check CPU offloading overhead
2. Reduce inference steps
3. Disable unnecessary optimizations (tiling has overhead)
4. Use FP16 instead of FP32

### CUDA Errors
**Symptom:** `RuntimeError: CUDA error: invalid configuration argument`

**Solutions:**
1. Update CUDA drivers
2. Restart runtime
3. Clear CUDA cache: `torch.cuda.empty_cache()`

---

## 📊 Performance Metrics

### Typical Performance (SD 1.5, 512x512, 50 steps)
| VRAM | Config | Time | Max Batch |
|------|--------|------|-----------|
| 4GB  | FP16 + Tiling | ~30s | 1 |
| 6GB  | FP16 + Slicing | ~20s | 2 |
| 8GB  | FP16 | ~15s | 4 |
| 12GB | FP16 | ~10s | 8 |

### Memory Overhead
- Base model (SD 1.5): ~3.5GB
- Text encoder: ~0.5GB
- UNet: ~1.5GB
- VAE: ~0.3GB
- Activations: ~1-2GB (depends on size)
- **Total with offloading:** ~4-5GB peak

---

## 🔮 Future Enhancements

- [ ] Model caching untuk multiple models
- [ ] Distributed inference support
- [ ] LoRA/ControlNet integration
- [ ] Video generation support
- [ ] Advanced scheduling algorithms
- [ ] Metrics dashboard
- [ ] API server mode
- [ ] Docker containerization

---

## 📄 License

MIT License - lihat file LICENSE

## 🤝 Contributing

Contributions welcome! Silakan buat PR atau issue.

---

## 📞 Support

Jika ada pertanyaan atau issues:
1. Check troubleshooting section
2. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
3. Check memory report: `runtime.get_stats()`
4. Review execution metrics dalam result

---

**Built for Google Colab and memory-constrained environments** 🚀
