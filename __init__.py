"""
Diffusion Runtime - Fault-Tolerant Inference System

A production-ready diffusion inference runtime optimized for memory-constrained
environments like Google Colab. Features phase-based model loading, automatic
OOM recovery, job queue system, and comprehensive monitoring.

Usage:
    from diffusion_runtime import DiffusionRuntime
    from diffusers import StableDiffusionPipeline
    
    runtime = DiffusionRuntime(model_id="runwayml/stable-diffusion-v1-5")
    runtime.start(StableDiffusionPipeline)
    
    result = runtime.generate_sync(prompt="A beautiful landscape")
    result.result.images[0].save("output.png")
    
    runtime.stop()
"""

from .src.core import (
    # Memory Management
    VRAMMonitor,
    MemoryManager,
    MemoryOptimizer,
    
    # Model Loading
    DiffusionModelLoader,
    ModelConfig,
    
    # Execution
    ExecutionStateMachine,
    ExecutionContext,
    ExecutionConfig,
    
    # Job Queue
    JobManager,
    JobRequest,
    JobPriority,
    JobStatus,
    
    # Inference Engine
    DiffusionInferenceEngine,
    InferenceResult
)

from .src.config import (
    RuntimeConfig,
    PresetConfigs,
    detect_environment
)

from .src.utils import (
    MemoryMonitor,
    PerformanceBenchmark,
    DebugLogger
)

# Import main runtime from examples (for convenience)
import sys
from pathlib import Path
examples_path = Path(__file__).parent / "examples"
sys.path.insert(0, str(examples_path))

try:
    from example_usage import DiffusionRuntime
except ImportError:
    # If examples not available, that's ok
    pass

__version__ = "1.0.0"
__author__ = "Diffusion Runtime Team"
__license__ = "MIT"

__all__ = [
    # Main Runtime
    "DiffusionRuntime",
    
    # Core Components
    "VRAMMonitor",
    "MemoryManager",
    "MemoryOptimizer",
    "DiffusionModelLoader",
    "ModelConfig",
    "ExecutionStateMachine",
    "ExecutionContext",
    "ExecutionConfig",
    "JobManager",
    "JobRequest",
    "JobPriority",
    "JobStatus",
    "DiffusionInferenceEngine",
    "InferenceResult",
    
    # Configuration
    "RuntimeConfig",
    "PresetConfigs",
    "detect_environment",
    
    # Utilities
    "MemoryMonitor",
    "PerformanceBenchmark",
    "DebugLogger",
]
