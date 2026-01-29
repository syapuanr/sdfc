"""
Job Queue Manager untuk Diffusion Runtime
Menangani job scheduling, priority queue, dan concurrent execution control
"""

import logging
import threading
import queue
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

from execution_state_machine import ExecutionContext, ExecutionConfig

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Priority levels untuk jobs"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class JobStatus(Enum):
    """Status untuk individual jobs"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY_PENDING = "retry_pending"


@dataclass
class JobRequest:
    """Request untuk inference job"""
    # Unique identifier
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Priority
    priority: JobPriority = JobPriority.NORMAL
    
    # Generation parameters
    prompt: str = ""
    negative_prompt: str = ""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    batch_size: int = 1
    seed: Optional[int] = None
    
    # Advanced parameters
    scheduler_type: str = "DPMSolverMultistep"
    enable_tiling: bool = False
    enable_attention_slicing: bool = True
    
    # Execution config
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    
    # Callbacks
    on_progress: Optional[Callable[[int, int], None]] = None
    on_complete: Optional[Callable[[Any], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None
    
    def to_execution_context(self) -> ExecutionContext:
        """Convert job request ke execution context"""
        return ExecutionContext(
            job_id=self.job_id,
            config=self.execution_config,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=self.height,
            width=self.width,
            batch_size=self.batch_size
        )
    
    def __lt__(self, other):
        """Comparison untuk priority queue (higher priority = lower number)"""
        # Invert priority sehingga higher priority comes first
        return self.priority.value > other.priority.value


@dataclass
class JobResult:
    """Result dari completed job"""
    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[Exception] = None
    error_message: Optional[str] = None
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Metrics
    retry_count: int = 0
    peak_vram_gb: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "has_result": self.result is not None,
            "has_error": self.error is not None,
            "error_message": self.error_message,
            "timing": {
                "start": self.start_time.isoformat() if self.start_time else None,
                "end": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": self.duration_seconds
            },
            "metrics": {
                "retry_count": self.retry_count,
                "peak_vram_gb": self.peak_vram_gb
            }
        }


class JobQueue:
    """
    Priority queue untuk jobs
    Thread-safe dengan blocking operations
    """
    
    def __init__(self, maxsize: int = 0):
        """
        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self.queue = queue.PriorityQueue(maxsize=maxsize)
        self.active_jobs: Dict[str, JobRequest] = {}
        self.completed_jobs: Dict[str, JobResult] = {}
        self.lock = threading.Lock()
        
        # Stats
        self.total_queued = 0
        self.total_completed = 0
        self.total_failed = 0
        
        logger.info(f"JobQueue initialized (maxsize={maxsize})")
    
    def enqueue(self, job: JobRequest, block: bool = True, timeout: Optional[float] = None):
        """
        Add job ke queue
        
        Args:
            job: Job request to enqueue
            block: Block if queue is full
            timeout: Timeout for blocking
        """
        try:
            self.queue.put(job, block=block, timeout=timeout)
            
            with self.lock:
                self.total_queued += 1
            
            logger.info(
                f"Job {job.job_id} enqueued (priority={job.priority.name}, "
                f"queue_size={self.queue.qsize()})"
            )
            
        except queue.Full:
            logger.error(f"Queue is full, cannot enqueue job {job.job_id}")
            raise
    
    def dequeue(self, block: bool = True, timeout: Optional[float] = None) -> JobRequest:
        """
        Get next job from queue
        
        Args:
            block: Block if queue is empty
            timeout: Timeout for blocking
        
        Returns:
            Next job request
        """
        try:
            job = self.queue.get(block=block, timeout=timeout)
            
            with self.lock:
                self.active_jobs[job.job_id] = job
            
            logger.info(
                f"Job {job.job_id} dequeued (queue_size={self.queue.qsize()})"
            )
            
            return job
            
        except queue.Empty:
            raise queue.Empty("No jobs in queue")
    
    def mark_completed(self, job_id: str, result: JobResult):
        """Mark job as completed"""
        with self.lock:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            self.completed_jobs[job_id] = result
            
            if result.status == JobStatus.COMPLETED:
                self.total_completed += 1
            elif result.status == JobStatus.FAILED:
                self.total_failed += 1
        
        logger.info(
            f"Job {job_id} marked as {result.status.value} "
            f"(duration={result.duration_seconds:.2f}s)"
        )
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel queued job
        
        Returns:
            True if job was cancelled, False if not found or already running
        """
        # Note: This doesn't work well with PriorityQueue
        # Would need custom implementation for proper cancellation
        logger.warning(f"Job cancellation requested for {job_id} (not implemented)")
        return False
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get status of specific job"""
        with self.lock:
            if job_id in self.active_jobs:
                return JobStatus.RUNNING
            if job_id in self.completed_jobs:
                return self.completed_jobs[job_id].status
        
        # Check if still in queue (expensive operation)
        # Would need better tracking for this
        return None
    
    def get_stats(self) -> Dict:
        """Get queue statistics"""
        with self.lock:
            return {
                "queue_size": self.queue.qsize(),
                "active_jobs": len(self.active_jobs),
                "completed_jobs": len(self.completed_jobs),
                "total_queued": self.total_queued,
                "total_completed": self.total_completed,
                "total_failed": self.total_failed,
                "success_rate": (
                    self.total_completed / (self.total_completed + self.total_failed)
                    if (self.total_completed + self.total_failed) > 0
                    else 0.0
                )
            }
    
    def clear(self):
        """Clear all queued jobs"""
        with self.lock:
            # Clear the queue
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            
            logger.info("Queue cleared")


class JobScheduler:
    """
    Scheduler untuk managing job execution
    Handles worker threads dan job dispatching
    """
    
    def __init__(self, 
                 job_queue: JobQueue,
                 executor_fn: Callable[[JobRequest], JobResult],
                 num_workers: int = 1):
        """
        Args:
            job_queue: Job queue to process
            executor_fn: Function yang menjalankan job dan return JobResult
            num_workers: Number of worker threads (biasanya 1 untuk GPU)
        """
        self.job_queue = job_queue
        self.executor_fn = executor_fn
        self.num_workers = num_workers
        
        self.workers: List[threading.Thread] = []
        self.running = False
        self.shutdown_event = threading.Event()
        
        logger.info(f"JobScheduler initialized with {num_workers} workers")
    
    def start(self):
        """Start scheduler dan worker threads"""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            logger.info(f"Started worker thread: Worker-{i}")
        
        logger.info("Scheduler started")
    
    def stop(self, timeout: float = 30.0):
        """Stop scheduler dan wait untuk workers"""
        if not self.running:
            logger.warning("Scheduler not running")
            return
        
        logger.info("Stopping scheduler...")
        self.running = False
        self.shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
            if worker.is_alive():
                logger.warning(f"Worker {worker.name} did not stop gracefully")
        
        self.workers.clear()
        logger.info("Scheduler stopped")
    
    def _worker_loop(self):
        """Main worker loop"""
        worker_name = threading.current_thread().name
        logger.info(f"{worker_name} started")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Get next job (with timeout sehingga bisa check shutdown)
                job = self.job_queue.dequeue(block=True, timeout=1.0)
                
                logger.info(f"{worker_name} processing job {job.job_id}")
                
                # Execute job
                start_time = datetime.now()
                result = self.executor_fn(job)
                end_time = datetime.now()
                
                # Update timing
                result.start_time = start_time
                result.end_time = end_time
                result.duration_seconds = (end_time - start_time).total_seconds()
                
                # Mark as completed
                self.job_queue.mark_completed(job.job_id, result)
                
                # Call completion callback if provided
                if job.on_complete and result.status == JobStatus.COMPLETED:
                    try:
                        job.on_complete(result.result)
                    except Exception as e:
                        logger.error(f"Completion callback failed: {e}")
                
                # Call error callback if provided
                if job.on_error and result.status == JobStatus.FAILED:
                    try:
                        job.on_error(result.error)
                    except Exception as e:
                        logger.error(f"Error callback failed: {e}")
                
            except queue.Empty:
                # No jobs available, continue loop
                continue
            except Exception as e:
                logger.error(f"{worker_name} encountered error: {e}", exc_info=True)
                # Continue processing
                continue
        
        logger.info(f"{worker_name} stopped")


class JobManager:
    """
    High-level manager combining queue dan scheduler
    """
    
    def __init__(self, 
                 executor_fn: Callable[[JobRequest], JobResult],
                 max_queue_size: int = 100,
                 num_workers: int = 1):
        """
        Args:
            executor_fn: Function untuk execute jobs
            max_queue_size: Maximum queue size
            num_workers: Number of worker threads
        """
        self.queue = JobQueue(maxsize=max_queue_size)
        self.scheduler = JobScheduler(self.queue, executor_fn, num_workers)
        
        logger.info("JobManager initialized")
    
    def start(self):
        """Start job processing"""
        self.scheduler.start()
    
    def stop(self, timeout: float = 30.0):
        """Stop job processing"""
        self.scheduler.stop(timeout)
    
    def submit_job(self, job: JobRequest) -> str:
        """
        Submit job for processing
        
        Returns:
            Job ID
        """
        self.queue.enqueue(job)
        return job.job_id
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get status of job"""
        return self.queue.get_job_status(job_id)
    
    def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """Get result of completed job"""
        return self.queue.completed_jobs.get(job_id)
    
    def get_stats(self) -> Dict:
        """Get queue statistics"""
        return self.queue.get_stats()
    
    def clear_queue(self):
        """Clear all queued jobs"""
        self.queue.clear()


if __name__ == "__main__":
    # Example usage
    print("Job Queue Manager Test")
    print("=" * 50)
    
    # Mock executor function
    def mock_executor(job: JobRequest) -> JobResult:
        """Mock executor untuk testing"""
        print(f"Executing job {job.job_id}: {job.prompt[:50]}")
        
        # Simulate work
        time.sleep(1)
        
        # Return success
        return JobResult(
            job_id=job.job_id,
            status=JobStatus.COMPLETED,
            result={"image": "fake_image_data"}
        )
    
    # Create manager
    manager = JobManager(
        executor_fn=mock_executor,
        max_queue_size=10,
        num_workers=1
    )
    
    # Start processing
    manager.start()
    
    print("\nSubmitting test jobs...")
    
    # Submit some test jobs
    job_ids = []
    for i in range(3):
        job = JobRequest(
            prompt=f"Test prompt {i}",
            priority=JobPriority.NORMAL if i % 2 == 0 else JobPriority.HIGH,
            num_inference_steps=20
        )
        job_id = manager.submit_job(job)
        job_ids.append(job_id)
        print(f"Submitted job {job_id}")
    
    # Show stats
    print(f"\nQueue stats: {manager.get_stats()}")
    
    # Wait for completion
    print("\nWaiting for jobs to complete...")
    time.sleep(5)
    
    # Check results
    print("\nJob results:")
    for job_id in job_ids:
        result = manager.get_job_result(job_id)
        if result:
            print(f"  {job_id}: {result.status.value}")
    
    # Final stats
    print(f"\nFinal stats: {manager.get_stats()}")
    
    # Stop manager
    print("\nStopping manager...")
    manager.stop()
    print("Done!")
