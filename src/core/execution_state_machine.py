"""
Execution State Machine untuk Diffusion Runtime
Menangani state transitions, retry logic, dan error recovery
"""

import torch
import logging
from enum import Enum, auto
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
import traceback

from memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    """States dalam execution lifecycle"""
    IDLE = auto()
    QUEUED = auto()
    INITIALIZING = auto()
    LOADING_TEXT_ENCODER = auto()
    ENCODING_PROMPT = auto()
    LOADING_UNET = auto()
    DIFFUSION_RUNNING = auto()
    LOADING_VAE = auto()
    DECODING = auto()
    POST_PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRY_PENDING = auto()
    CANCELLED = auto()


class ErrorType(Enum):
    """Tipe error yang bisa terjadi"""
    OOM_ERROR = "out_of_memory"
    CUDA_ERROR = "cuda_error"
    MODEL_LOAD_ERROR = "model_load_error"
    INFERENCE_ERROR = "inference_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ExecutionMetrics:
    """Metrics untuk satu execution"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Phase timings
    encoding_time: float = 0.0
    diffusion_time: float = 0.0
    decoding_time: float = 0.0
    total_time: float = 0.0
    
    # Memory stats
    peak_vram_gb: float = 0.0
    avg_vram_gb: float = 0.0
    
    # Retry stats
    retry_count: int = 0
    oom_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "timings": {
                "encoding": self.encoding_time,
                "diffusion": self.diffusion_time,
                "decoding": self.decoding_time,
                "total": self.total_time
            },
            "memory": {
                "peak_vram_gb": self.peak_vram_gb,
                "avg_vram_gb": self.avg_vram_gb
            },
            "retries": {
                "total": self.retry_count,
                "oom": self.oom_count
            }
        }


@dataclass
class ExecutionConfig:
    """Configuration untuk execution behavior"""
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    enable_progressive_degradation: bool = True  # Reduce quality on retry
    timeout_seconds: Optional[float] = 300  # 5 minutes default
    
    # OOM handling strategies
    reduce_batch_size_on_oom: bool = True
    enable_tiling_on_oom: bool = True
    reduce_resolution_on_oom: bool = False  # Last resort
    
    # Checkpointing
    enable_intermediate_save: bool = True
    checkpoint_interval: int = 10  # Save every N steps


@dataclass
class ExecutionContext:
    """Context untuk satu execution job"""
    job_id: str
    state: ExecutionState = ExecutionState.IDLE
    config: ExecutionConfig = field(default_factory=ExecutionConfig)
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    
    # Job parameters
    prompt: str = ""
    negative_prompt: str = ""
    num_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    batch_size: int = 1
    
    # Results
    result: Optional[Any] = None
    error: Optional[Exception] = None
    error_type: Optional[ErrorType] = None
    error_traceback: Optional[str] = None
    
    # State tracking
    current_step: int = 0
    retry_attempt: int = 0
    intermediate_results: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Serialize context untuk logging/monitoring"""
        return {
            "job_id": self.job_id,
            "state": self.state.name,
            "parameters": {
                "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
                "num_steps": self.num_steps,
                "guidance_scale": self.guidance_scale,
                "size": f"{self.height}x{self.width}",
                "batch_size": self.batch_size
            },
            "progress": {
                "current_step": self.current_step,
                "total_steps": self.num_steps,
                "percent": (self.current_step / self.num_steps * 100) if self.num_steps > 0 else 0
            },
            "retry_attempt": self.retry_attempt,
            "has_error": self.error is not None,
            "error_type": self.error_type.value if self.error_type else None,
            "metrics": self.metrics.to_dict()
        }


class OOMRecoveryStrategy:
    """Strategies untuk recovery dari OOM errors"""
    
    @staticmethod
    def reduce_batch_size(context: ExecutionContext) -> ExecutionContext:
        """Reduce batch size"""
        if context.batch_size > 1:
            old_batch = context.batch_size
            context.batch_size = max(1, context.batch_size // 2)
            logger.info(f"Reduced batch size: {old_batch} → {context.batch_size}")
        return context
    
    @staticmethod
    def enable_more_tiling(context: ExecutionContext) -> ExecutionContext:
        """Enable atau increase tiling"""
        # This would be implemented by the actual execution engine
        logger.info("Enabling aggressive tiling for VAE decode")
        return context
    
    @staticmethod
    def reduce_resolution(context: ExecutionContext) -> ExecutionContext:
        """Reduce resolution (last resort)"""
        scale_factor = 0.75  # Reduce by 25%
        old_h, old_w = context.height, context.width
        context.height = int(context.height * scale_factor)
        context.width = int(context.width * scale_factor)
        
        # Round to nearest multiple of 8 (requirement untuk VAE)
        context.height = (context.height // 8) * 8
        context.width = (context.width // 8) * 8
        
        logger.warning(
            f"Reduced resolution: {old_h}x{old_w} → {context.height}x{context.width}"
        )
        return context
    
    @staticmethod
    def progressive_degradation(context: ExecutionContext) -> ExecutionContext:
        """Apply progressive degradation based on retry count"""
        if context.retry_attempt == 1:
            # First retry: reduce batch size
            return OOMRecoveryStrategy.reduce_batch_size(context)
        elif context.retry_attempt == 2:
            # Second retry: enable tiling
            return OOMRecoveryStrategy.enable_more_tiling(context)
        elif context.retry_attempt >= 3:
            # Last resort: reduce resolution
            return OOMRecoveryStrategy.reduce_resolution(context)
        return context


class ExecutionStateMachine:
    """
    State machine untuk mengelola execution lifecycle
    Handles transitions, errors, dan retry logic
    """
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.current_context: Optional[ExecutionContext] = None
        
        # State transition handlers
        self.transition_handlers: Dict[ExecutionState, Callable] = {}
        
        # Error handlers
        self.error_handlers: Dict[ErrorType, Callable] = {
            ErrorType.OOM_ERROR: self._handle_oom_error,
            ErrorType.CUDA_ERROR: self._handle_cuda_error,
            ErrorType.MODEL_LOAD_ERROR: self._handle_model_load_error,
        }
        
        logger.info("ExecutionStateMachine initialized")
    
    def start_execution(self, context: ExecutionContext) -> ExecutionContext:
        """Start new execution dengan given context"""
        if self.current_context and self.current_context.state not in [
            ExecutionState.COMPLETED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED
        ]:
            raise RuntimeError(
                f"Cannot start new execution. Current job {self.current_context.job_id} "
                f"is in state {self.current_context.state.name}"
            )
        
        self.current_context = context
        context.metrics.start_time = datetime.now()
        
        logger.info(f"Starting execution for job {context.job_id}")
        logger.info(f"Parameters: {context.to_dict()['parameters']}")
        
        return self._transition_to(ExecutionState.INITIALIZING)
    
    def _transition_to(self, new_state: ExecutionState) -> ExecutionContext:
        """Transition ke state baru"""
        if not self.current_context:
            raise RuntimeError("No active execution context")
        
        old_state = self.current_context.state
        self.current_context.state = new_state
        
        logger.info(f"State transition: {old_state.name} → {new_state.name}")
        
        # Call transition handler jika ada
        if new_state in self.transition_handlers:
            try:
                self.transition_handlers[new_state](self.current_context)
            except Exception as e:
                logger.error(f"Transition handler failed: {e}")
                self._handle_error(e)
        
        return self.current_context
    
    def _handle_error(self, error: Exception) -> ExecutionContext:
        """Handle error yang terjadi during execution"""
        if not self.current_context:
            raise RuntimeError("No active execution context")
        
        context = self.current_context
        context.error = error
        context.error_traceback = traceback.format_exc()
        
        # Classify error type
        error_type = self._classify_error(error)
        context.error_type = error_type
        
        logger.error(f"Execution error ({error_type.value}): {error}")
        logger.debug(f"Traceback: {context.error_traceback}")
        
        # Update metrics
        if error_type == ErrorType.OOM_ERROR:
            context.metrics.oom_count += 1
        
        # Try to recover
        can_retry = context.retry_attempt < context.config.max_retries
        
        if can_retry:
            logger.info(
                f"Attempting recovery (retry {context.retry_attempt + 1}/"
                f"{context.config.max_retries})"
            )
            
            # Call appropriate error handler
            if error_type in self.error_handlers:
                try:
                    self.error_handlers[error_type](context)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {handler_error}")
                    return self._transition_to(ExecutionState.FAILED)
            
            # Increment retry counter
            context.retry_attempt += 1
            context.metrics.retry_count += 1
            
            # Apply progressive degradation jika enabled
            if context.config.enable_progressive_degradation:
                OOMRecoveryStrategy.progressive_degradation(context)
            
            # Transition to retry pending
            return self._transition_to(ExecutionState.RETRY_PENDING)
        else:
            logger.error(
                f"Max retries ({context.config.max_retries}) exceeded. "
                f"Execution failed."
            )
            return self._transition_to(ExecutionState.FAILED)
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type dari exception"""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # OOM detection
        if "out of memory" in error_str or "oom" in error_str:
            return ErrorType.OOM_ERROR
        
        # CUDA errors
        if "cuda" in error_str or "cudnn" in error_str:
            return ErrorType.CUDA_ERROR
        
        # Model loading errors
        if "load" in error_str or "checkpoint" in error_str:
            return ErrorType.MODEL_LOAD_ERROR
        
        # Timeout
        if "timeout" in error_str:
            return ErrorType.TIMEOUT_ERROR
        
        return ErrorType.UNKNOWN_ERROR
    
    def _handle_oom_error(self, context: ExecutionContext):
        """Handle OOM error"""
        logger.warning("Handling OOM error...")
        
        # Aggressive memory cleanup
        self.memory_manager.aggressive_cleanup()
        
        # Log memory status
        self.memory_manager.monitor.log_memory_status("After OOM cleanup - ")
        
        # Apply OOM-specific strategies
        if context.config.reduce_batch_size_on_oom:
            OOMRecoveryStrategy.reduce_batch_size(context)
        
        if context.config.enable_tiling_on_oom:
            OOMRecoveryStrategy.enable_more_tiling(context)
    
    def _handle_cuda_error(self, context: ExecutionContext):
        """Handle CUDA error"""
        logger.warning("Handling CUDA error...")
        
        # Reset CUDA context if possible
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.info("CUDA context reset successful")
        except Exception as e:
            logger.error(f"Failed to reset CUDA context: {e}")
    
    def _handle_model_load_error(self, context: ExecutionContext):
        """Handle model loading error"""
        logger.warning("Handling model load error...")
        
        # Clear cache and retry
        self.memory_manager.aggressive_cleanup()
    
    def complete_execution(self, result: Any) -> ExecutionContext:
        """Mark execution as completed"""
        if not self.current_context:
            raise RuntimeError("No active execution context")
        
        context = self.current_context
        context.result = result
        context.metrics.end_time = datetime.now()
        
        # Calculate total time
        if context.metrics.start_time:
            delta = context.metrics.end_time - context.metrics.start_time
            context.metrics.total_time = delta.total_seconds()
        
        logger.info(f"Execution completed for job {context.job_id}")
        logger.info(f"Total time: {context.metrics.total_time:.2f}s")
        logger.info(f"Retries: {context.metrics.retry_count}")
        
        return self._transition_to(ExecutionState.COMPLETED)
    
    def cancel_execution(self) -> ExecutionContext:
        """Cancel current execution"""
        if not self.current_context:
            raise RuntimeError("No active execution context")
        
        logger.info(f"Cancelling execution for job {self.current_context.job_id}")
        return self._transition_to(ExecutionState.CANCELLED)
    
    def get_status(self) -> Dict:
        """Get current execution status"""
        if not self.current_context:
            return {
                "status": "IDLE",
                "has_active_job": False
            }
        
        return {
            "status": self.current_context.state.name,
            "has_active_job": True,
            "context": self.current_context.to_dict()
        }


class ExecutionMonitor:
    """
    Monitor untuk tracking execution progress dan health
    """
    
    def __init__(self, state_machine: ExecutionStateMachine, memory_manager: MemoryManager):
        self.state_machine = state_machine
        self.memory_manager = memory_manager
        self.vram_samples: List[float] = []
    
    def update_progress(self, current_step: int):
        """Update progress untuk current step"""
        context = self.state_machine.current_context
        if context:
            context.current_step = current_step
            
            # Sample VRAM
            stats = self.memory_manager.monitor.get_memory_stats()
            self.vram_samples.append(stats.allocated_gb)
            
            # Update peak
            context.metrics.peak_vram_gb = max(
                context.metrics.peak_vram_gb,
                stats.allocated_gb
            )
            
            # Log progress periodically
            if current_step % 10 == 0 or current_step == context.num_steps:
                logger.info(
                    f"Step {current_step}/{context.num_steps} "
                    f"({current_step/context.num_steps*100:.1f}%) - "
                    f"VRAM: {stats.allocated_gb:.2f}GB"
                )
    
    def finalize_metrics(self):
        """Finalize metrics at end of execution"""
        context = self.state_machine.current_context
        if context and self.vram_samples:
            context.metrics.avg_vram_gb = sum(self.vram_samples) / len(self.vram_samples)
        
        self.vram_samples.clear()
    
    def check_health(self) -> bool:
        """Check execution health"""
        # Check if memory is critical
        if self.memory_manager.monitor.is_critical():
            logger.warning("CRITICAL: VRAM usage is critically high")
            return False
        
        # Check timeout if configured
        context = self.state_machine.current_context
        if context and context.config.timeout_seconds:
            elapsed = (datetime.now() - context.metrics.start_time).total_seconds()
            if elapsed > context.config.timeout_seconds:
                logger.error(f"TIMEOUT: Execution exceeded {context.config.timeout_seconds}s")
                return False
        
        return True


if __name__ == "__main__":
    # Example usage
    print("Execution State Machine Test")
    print("=" * 50)
    
    # Setup
    memory_manager = MemoryManager()
    state_machine = ExecutionStateMachine(memory_manager)
    monitor = ExecutionMonitor(state_machine, memory_manager)
    
    # Create test context
    context = ExecutionContext(
        job_id="test-job-001",
        prompt="A beautiful landscape",
        num_steps=50,
        height=512,
        width=512
    )
    
    # Start execution
    print("\nStarting execution...")
    state_machine.start_execution(context)
    print(f"State: {state_machine.get_status()['status']}")
    
    # Simulate progress
    print("\nSimulating progress...")
    for step in range(0, 51, 10):
        monitor.update_progress(step)
    
    # Complete
    print("\nCompleting execution...")
    state_machine.complete_execution(result={"image": "fake_image_data"})
    
    # Show final status
    status = state_machine.get_status()
    print(f"\nFinal state: {status['status']}")
    print(f"Metrics: {status['context']['metrics']}")
