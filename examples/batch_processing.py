"""
Batch Processing Example
Contoh untuk batch processing multiple images dengan job queue
"""

import torch
from diffusers import StableDiffusionPipeline
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import JobManager, JobRequest, JobPriority, JobStatus, DiffusionInferenceEngine
from src.config import detect_environment


def execute_job(job_request):
    """Execute single job"""
    from src.core import ExecutionContext, JobResult
    
    # Create engine (simplified - in production, reuse engine)
    engine = DiffusionInferenceEngine(
        model_id="runwayml/stable-diffusion-v1-5",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    try:
        # Initialize if needed
        if not engine.is_initialized:
            engine.initialize(StableDiffusionPipeline)
        
        # Generate
        result = engine.generate(
            prompt=job_request.prompt,
            negative_prompt=job_request.negative_prompt,
            num_inference_steps=job_request.num_inference_steps,
            height=job_request.height,
            width=job_request.width,
            num_images=job_request.batch_size,
            seed=job_request.seed
        )
        
        # Return success
        return JobResult(
            job_id=job_request.job_id,
            status=JobStatus.COMPLETED,
            result=result
        )
        
    except Exception as e:
        return JobResult(
            job_id=job_request.job_id,
            status=JobStatus.FAILED,
            error=e,
            error_message=str(e)
        )


def main():
    """Batch processing example"""
    print("=" * 70)
    print("Batch Processing Example")
    print("=" * 70)
    
    # Create job manager
    job_manager = JobManager(
        executor_fn=execute_job,
        max_queue_size=100,
        num_workers=1
    )
    
    # Start processing
    job_manager.start()
    
    # Define prompts
    prompts = [
        ("A cat sitting on a table", JobPriority.HIGH),
        ("A dog running in a park", JobPriority.NORMAL),
        ("A bird flying in the sky", JobPriority.NORMAL),
        ("A fish swimming in ocean", JobPriority.LOW),
    ]
    
    print(f"\nSubmitting {len(prompts)} jobs...")
    job_ids = []
    
    for prompt, priority in prompts:
        job = JobRequest(
            prompt=prompt,
            negative_prompt="blurry, low quality",
            num_inference_steps=25,
            height=512,
            width=512,
            priority=priority
        )
        
        job_id = job_manager.submit_job(job)
        job_ids.append((job_id, prompt))
        print(f"  ✓ {prompt[:40]}... (Priority: {priority.name})")
    
    # Wait for completion
    print("\nProcessing jobs...")
    while True:
        stats = job_manager.get_stats()
        if stats['queue_size'] == 0 and stats['active_jobs'] == 0:
            break
        
        print(f"  Queue: {stats['queue_size']}, Active: {stats['active_jobs']}", end='\r')
        time.sleep(1)
    
    print("\n\nAll jobs completed!")
    
    # Show results
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    
    output_dir = Path("outputs/batch")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (job_id, prompt) in enumerate(job_ids):
        result = job_manager.get_job_result(job_id)
        
        print(f"\n{i+1}. {prompt[:50]}")
        print(f"   Status: {result.status.value}")
        
        if result.status == JobStatus.COMPLETED and result.result:
            output_path = output_dir / f"batch_{i+1}.png"
            result.result.images[0].save(output_path)
            print(f"   Saved: {output_path}")
            print(f"   Time: {result.duration_seconds:.2f}s")
    
    # Final stats
    stats = job_manager.get_stats()
    print("\n" + "=" * 70)
    print("Final Statistics:")
    print("=" * 70)
    print(f"Total completed: {stats['total_completed']}")
    print(f"Total failed: {stats['total_failed']}")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")
    
    # Stop manager
    job_manager.stop()
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
