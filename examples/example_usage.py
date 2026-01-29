"""
Complete Example: Fault-Tolerant Diffusion Runtime dengan Job Queue
Demonstrasi penggunaan semua komponen secara terintegrasi
"""

import torch
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('diffusion_runtime.log')
    ]
)
logger = logging.getLogger(__name__)

from diffusion_engine import DiffusionInferenceEngine
from job_queue_manager import JobManager, JobRequest, JobResult, JobStatus, JobPriority
from model_loader import ModelConfig


class DiffusionRuntime:
    """
    Complete runtime system dengan job queue dan fault tolerance
    """
    
    def __init__(self,
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "cuda",
                 max_queue_size: int = 100):
        """
        Args:
            model_id: Hugging Face model ID
            device: Device untuk inference
            max_queue_size: Maximum job queue size
        """
        self.model_id = model_id
        self.device = device
        
        # Initialize engine
        logger.info("Initializing Diffusion Runtime...")
        self.engine = DiffusionInferenceEngine(
            model_id=model_id,
            device=device,
            enable_cpu_offload=True
        )
        
        # Initialize job manager dengan executor
        self.job_manager = JobManager(
            executor_fn=self._execute_job,
            max_queue_size=max_queue_size,
            num_workers=1  # Single GPU = 1 worker
        )
        
        self.is_running = False
        logger.info("Runtime initialized")
    
    def start(self, pipeline_class):
        """
        Start runtime
        
        Args:
            pipeline_class: Diffusers pipeline class (e.g., StableDiffusionPipeline)
        """
        if self.is_running:
            logger.warning("Runtime already running")
            return
        
        logger.info("=" * 70)
        logger.info("STARTING DIFFUSION RUNTIME")
        logger.info("=" * 70)
        
        # Initialize engine
        self.engine.initialize(pipeline_class)
        
        # Start job processing
        self.job_manager.start()
        
        self.is_running = True
        logger.info("Runtime started successfully")
        logger.info("Ready to accept jobs")
        
        # Show memory status
        report = self.engine.get_memory_report()
        logger.info(f"Initial VRAM: {report['vram']['allocated_gb']:.2f}GB / "
                   f"{report['vram']['total_gb']:.2f}GB")
    
    def stop(self):
        """Stop runtime"""
        if not self.is_running:
            logger.warning("Runtime not running")
            return
        
        logger.info("Stopping runtime...")
        
        # Stop job processing
        self.job_manager.stop(timeout=60.0)
        
        # Cleanup engine
        self.engine.cleanup()
        
        self.is_running = False
        logger.info("Runtime stopped")
    
    def _execute_job(self, job: JobRequest) -> JobResult:
        """
        Execute single job (called by job manager)
        
        Args:
            job: Job request to execute
        
        Returns:
            Job result
        """
        logger.info("=" * 70)
        logger.info(f"EXECUTING JOB: {job.job_id}")
        logger.info("=" * 70)
        
        try:
            # Generate image
            result = self.engine.generate(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                num_inference_steps=job.num_inference_steps,
                guidance_scale=job.guidance_scale,
                height=job.height,
                width=job.width,
                num_images=job.batch_size,
                seed=job.seed
            )
            
            # Get metrics
            status = self.engine.state_machine.get_status()
            metrics = status['context']['metrics']
            
            # Create job result
            job_result = JobResult(
                job_id=job.job_id,
                status=JobStatus.COMPLETED,
                result=result,
                retry_count=metrics['retries']['total'],
                peak_vram_gb=metrics['memory']['peak_vram_gb']
            )
            
            logger.info(f"Job {job.job_id} completed successfully")
            return job_result
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            
            # Create failure result
            job_result = JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error=e,
                error_message=str(e)
            )
            
            return job_result
    
    def submit_job(self,
                   prompt: str,
                   negative_prompt: str = "",
                   num_steps: int = 50,
                   guidance_scale: float = 7.5,
                   height: int = 512,
                   width: int = 512,
                   batch_size: int = 1,
                   seed: int = None,
                   priority: JobPriority = JobPriority.NORMAL,
                   **kwargs) -> str:
        """
        Submit generation job
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            num_steps: Number of inference steps
            guidance_scale: Guidance scale
            height: Image height
            width: Image width
            batch_size: Number of images
            seed: Random seed
            priority: Job priority
            **kwargs: Additional job parameters
        
        Returns:
            Job ID
        """
        if not self.is_running:
            raise RuntimeError("Runtime not started. Call start() first.")
        
        # Create job request
        job = JobRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            batch_size=batch_size,
            seed=seed,
            priority=priority,
            **kwargs
        )
        
        # Submit to queue
        job_id = self.job_manager.submit_job(job)
        
        logger.info(f"Submitted job {job_id} (priority={priority.name})")
        logger.info(f"Queue size: {self.job_manager.get_stats()['queue_size']}")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of job"""
        return self.job_manager.get_job_status(job_id)
    
    def get_job_result(self, job_id: str) -> JobResult:
        """Get result of completed job"""
        return self.job_manager.get_job_result(job_id)
    
    def get_stats(self) -> dict:
        """Get runtime statistics"""
        queue_stats = self.job_manager.get_stats()
        memory_report = self.engine.get_memory_report()
        
        return {
            "queue": queue_stats,
            "memory": memory_report,
            "is_running": self.is_running
        }
    
    def generate_sync(self,
                      prompt: str,
                      **kwargs) -> JobResult:
        """
        Synchronous generation (blocks until complete)
        
        Args:
            prompt: Text prompt
            **kwargs: Generation parameters
        
        Returns:
            Job result with images
        """
        import time
        
        # Submit job
        job_id = self.submit_job(prompt, **kwargs)
        
        # Wait for completion
        logger.info(f"Waiting for job {job_id} to complete...")
        
        while True:
            status = self.get_job_status(job_id)
            
            if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                break
            
            time.sleep(0.5)
        
        # Get result
        result = self.get_job_result(job_id)
        
        if result.status == JobStatus.FAILED:
            raise RuntimeError(f"Job failed: {result.error_message}")
        
        return result


def example_usage():
    """Example: Basic usage"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Single Image Generation")
    print("=" * 70)
    
    # Import diffusers (must be installed)
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        print("ERROR: diffusers library not installed")
        print("Install with: pip install diffusers transformers accelerate")
        return
    
    # Create runtime
    runtime = DiffusionRuntime(
        model_id="runwayml/stable-diffusion-v1-5",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    try:
        # Start runtime
        runtime.start(StableDiffusionPipeline)
        
        # Generate image (synchronous)
        print("\nGenerating image...")
        result = runtime.generate_sync(
            prompt="A serene landscape with mountains and a lake at sunset, digital art",
            negative_prompt="blurry, ugly, low quality",
            num_steps=30,
            height=512,
            width=512,
            seed=42
        )
        
        # Save result
        if result.result and result.result.images:
            image = result.result.images[0]
            output_path = "output_example1.png"
            image.save(output_path)
            print(f"\nImage saved to: {output_path}")
            print(f"Generation time: {result.duration_seconds:.2f}s")
            print(f"Peak VRAM: {result.peak_vram_gb:.2f}GB")
            print(f"Retries: {result.retry_count}")
        
    finally:
        # Always cleanup
        runtime.stop()


def example_batch_jobs():
    """Example: Batch job processing"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Batch Job Processing with Priority Queue")
    print("=" * 70)
    
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        print("ERROR: diffusers library not installed")
        return
    
    # Create runtime
    runtime = DiffusionRuntime(
        model_id="runwayml/stable-diffusion-v1-5"
    )
    
    try:
        # Start runtime
        runtime.start(StableDiffusionPipeline)
        
        # Submit multiple jobs with different priorities
        prompts = [
            ("A cat sitting on a table", JobPriority.NORMAL),
            ("A dog running in a park", JobPriority.HIGH),
            ("A bird flying in the sky", JobPriority.LOW),
            ("A fish swimming in the ocean", JobPriority.URGENT),
        ]
        
        job_ids = []
        for prompt, priority in prompts:
            job_id = runtime.submit_job(
                prompt=prompt,
                num_steps=20,
                priority=priority
            )
            job_ids.append(job_id)
            print(f"Submitted: {prompt[:30]}... (priority={priority.name})")
        
        # Wait for all jobs to complete
        import time
        print("\nWaiting for jobs to complete...")
        
        completed = 0
        while completed < len(job_ids):
            time.sleep(1)
            
            # Check status
            completed = 0
            for job_id in job_ids:
                status = runtime.get_job_status(job_id)
                if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    completed += 1
            
            # Show progress
            stats = runtime.get_stats()
            print(f"Progress: {completed}/{len(job_ids)} completed, "
                  f"Queue: {stats['queue']['queue_size']}")
        
        # Show results
        print("\n" + "=" * 70)
        print("Results:")
        print("=" * 70)
        
        for i, job_id in enumerate(job_ids):
            result = runtime.get_job_result(job_id)
            print(f"\nJob {i+1}: {prompts[i][0][:40]}")
            print(f"  Status: {result.status.value}")
            print(f"  Duration: {result.duration_seconds:.2f}s")
            print(f"  Peak VRAM: {result.peak_vram_gb:.2f}GB")
            print(f"  Retries: {result.retry_count}")
            
            # Save image if successful
            if result.status == JobStatus.COMPLETED and result.result.images:
                output_path = f"output_batch_{i+1}.png"
                result.result.images[0].save(output_path)
                print(f"  Saved to: {output_path}")
        
        # Final stats
        print("\n" + "=" * 70)
        print("Final Statistics:")
        print("=" * 70)
        stats = runtime.get_stats()
        print(f"Total completed: {stats['queue']['total_completed']}")
        print(f"Total failed: {stats['queue']['total_failed']}")
        print(f"Success rate: {stats['queue']['success_rate']*100:.1f}%")
        
    finally:
        runtime.stop()


def example_error_recovery():
    """Example: Error recovery and retry logic"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Error Recovery with Progressive Degradation")
    print("=" * 70)
    
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        print("ERROR: diffusers library not installed")
        return
    
    runtime = DiffusionRuntime()
    
    try:
        runtime.start(StableDiffusionPipeline)
        
        # Try to generate with parameters that might cause OOM
        print("\nAttempting challenging generation...")
        print("(This might trigger OOM and retry with reduced parameters)")
        
        result = runtime.generate_sync(
            prompt="A highly detailed fantasy castle with intricate architecture",
            num_steps=50,
            height=768,  # Large size might cause OOM
            width=768,
            batch_size=2  # Multiple images increases memory
        )
        
        print("\n" + "=" * 70)
        print("Generation completed (possibly after retries)")
        print("=" * 70)
        print(f"Status: {result.status.value}")
        print(f"Retries performed: {result.retry_count}")
        print(f"Final parameters might have been adjusted for memory")
        
        if result.result and result.result.images:
            for i, image in enumerate(result.result.images):
                output_path = f"output_recovery_{i}.png"
                image.save(output_path)
                print(f"Saved image {i+1} to: {output_path}")
        
    except Exception as e:
        print(f"\nGeneration failed even after retries: {e}")
        print("This is expected if VRAM is too limited for the parameters")
    
    finally:
        runtime.stop()


if __name__ == "__main__":
    print("=" * 70)
    print("FAULT-TOLERANT DIFFUSION RUNTIME EXAMPLES")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. Examples require GPU.")
        print("The system will run but may be very slow on CPU.")
        input("Press Enter to continue anyway, or Ctrl+C to exit...")
    
    # Check memory
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {total_vram:.2f}GB")
        
        if total_vram < 8:
            print("\nWARNING: Less than 8GB VRAM detected.")
            print("Consider using smaller models or reduced parameters.")
    
    # Run examples
    print("\nSelect example to run:")
    print("1. Basic single image generation")
    print("2. Batch job processing")
    print("3. Error recovery demo")
    print("4. Run all examples")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            example_usage()
        elif choice == "2":
            example_batch_jobs()
        elif choice == "3":
            example_error_recovery()
        elif choice == "4":
            example_usage()
            example_batch_jobs()
            example_error_recovery()
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()
