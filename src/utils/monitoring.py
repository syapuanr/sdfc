"""
Utilities untuk monitoring, debugging, dan benchmarking
"""

import torch
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result dari benchmark run"""
    config: Dict[str, Any]
    num_runs: int
    
    # Timing
    avg_time: float
    min_time: float
    max_time: float
    std_time: float
    
    # Memory
    avg_vram_gb: float
    peak_vram_gb: float
    
    # Success rate
    successful_runs: int
    failed_runs: int
    oom_errors: int
    
    # Per-phase timing
    avg_encoding_time: float = 0.0
    avg_diffusion_time: float = 0.0
    avg_decoding_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "config": self.config,
            "num_runs": self.num_runs,
            "timing": {
                "avg": self.avg_time,
                "min": self.min_time,
                "max": self.max_time,
                "std": self.std_time,
                "phases": {
                    "encoding": self.avg_encoding_time,
                    "diffusion": self.avg_diffusion_time,
                    "decoding": self.avg_decoding_time
                }
            },
            "memory": {
                "avg_vram_gb": self.avg_vram_gb,
                "peak_vram_gb": self.peak_vram_gb
            },
            "reliability": {
                "successful": self.successful_runs,
                "failed": self.failed_runs,
                "oom_errors": self.oom_errors,
                "success_rate": self.successful_runs / self.num_runs if self.num_runs > 0 else 0
            }
        }


class MemoryMonitor:
    """
    Real-time memory monitoring utility
    """
    
    def __init__(self, device: str = "cuda:0", interval_seconds: float = 1.0):
        self.device = torch.device(device)
        self.interval = interval_seconds
        self.is_monitoring = False
        self.samples: List[Dict] = []
        
    def start(self):
        """Start monitoring in background thread"""
        import threading
        
        if self.is_monitoring:
            logger.warning("Already monitoring")
            return
        
        self.is_monitoring = True
        self.samples.clear()
        
        def monitor_loop():
            while self.is_monitoring:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(self.device) / 1e9
                    reserved = torch.cuda.memory_reserved(self.device) / 1e9
                    
                    self.samples.append({
                        "timestamp": time.time(),
                        "allocated_gb": allocated,
                        "reserved_gb": reserved
                    })
                
                time.sleep(self.interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop(self) -> Dict:
        """Stop monitoring and return statistics"""
        if not self.is_monitoring:
            logger.warning("Not monitoring")
            return {}
        
        self.is_monitoring = False
        self.monitor_thread.join(timeout=5)
        
        if not self.samples:
            return {}
        
        allocated = [s["allocated_gb"] for s in self.samples]
        
        stats = {
            "num_samples": len(self.samples),
            "duration_seconds": self.samples[-1]["timestamp"] - self.samples[0]["timestamp"],
            "avg_allocated_gb": sum(allocated) / len(allocated),
            "peak_allocated_gb": max(allocated),
            "min_allocated_gb": min(allocated),
        }
        
        logger.info(f"Memory monitoring stopped. Peak: {stats['peak_allocated_gb']:.2f}GB")
        return stats
    
    def get_samples(self) -> List[Dict]:
        """Get all memory samples"""
        return self.samples.copy()


class PerformanceBenchmark:
    """
    Benchmark utility untuk testing performance
    """
    
    def __init__(self, runtime):
        """
        Args:
            runtime: DiffusionRuntime instance
        """
        self.runtime = runtime
    
    def benchmark_config(self,
                        prompt: str,
                        num_runs: int = 5,
                        **generation_params) -> BenchmarkResult:
        """
        Benchmark specific configuration
        
        Args:
            prompt: Test prompt
            num_runs: Number of runs
            **generation_params: Parameters for generation
        
        Returns:
            BenchmarkResult
        """
        logger.info(f"Starting benchmark with {num_runs} runs")
        logger.info(f"Prompt: {prompt[:50]}...")
        logger.info(f"Params: {generation_params}")
        
        timings = []
        vram_peaks = []
        vram_avgs = []
        encoding_times = []
        diffusion_times = []
        decoding_times = []
        
        successful = 0
        failed = 0
        oom_errors = 0
        
        for i in range(num_runs):
            logger.info(f"\nRun {i+1}/{num_runs}")
            
            try:
                # Run generation
                result = self.runtime.generate_sync(
                    prompt=prompt,
                    **generation_params
                )
                
                # Collect metrics
                if result.status.value == "completed":
                    successful += 1
                    
                    # Get metrics from result
                    context = self.runtime.engine.state_machine.current_context
                    if context:
                        metrics = context.metrics
                        timings.append(metrics.total_time)
                        vram_peaks.append(metrics.peak_vram_gb)
                        vram_avgs.append(metrics.avg_vram_gb)
                        encoding_times.append(metrics.encoding_time)
                        diffusion_times.append(metrics.diffusion_time)
                        decoding_times.append(metrics.decoding_time)
                else:
                    failed += 1
                    if "oom" in str(result.error_message).lower():
                        oom_errors += 1
                
            except Exception as e:
                logger.error(f"Run {i+1} failed: {e}")
                failed += 1
                if "out of memory" in str(e).lower():
                    oom_errors += 1
        
        # Calculate statistics
        import numpy as np
        
        if timings:
            avg_time = np.mean(timings)
            min_time = np.min(timings)
            max_time = np.max(timings)
            std_time = np.std(timings)
            
            avg_vram = np.mean(vram_avgs) if vram_avgs else 0
            peak_vram = max(vram_peaks) if vram_peaks else 0
            
            avg_encoding = np.mean(encoding_times) if encoding_times else 0
            avg_diffusion = np.mean(diffusion_times) if diffusion_times else 0
            avg_decoding = np.mean(decoding_times) if decoding_times else 0
        else:
            avg_time = min_time = max_time = std_time = 0
            avg_vram = peak_vram = 0
            avg_encoding = avg_diffusion = avg_decoding = 0
        
        result = BenchmarkResult(
            config=generation_params,
            num_runs=num_runs,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_time=std_time,
            avg_vram_gb=avg_vram,
            peak_vram_gb=peak_vram,
            successful_runs=successful,
            failed_runs=failed,
            oom_errors=oom_errors,
            avg_encoding_time=avg_encoding,
            avg_diffusion_time=avg_diffusion,
            avg_decoding_time=avg_decoding
        )
        
        # Print summary
        self._print_benchmark_summary(result)
        
        return result
    
    def _print_benchmark_summary(self, result: BenchmarkResult):
        """Print benchmark summary"""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        
        print(f"\nRuns: {result.num_runs}")
        print(f"Successful: {result.successful_runs}")
        print(f"Failed: {result.failed_runs}")
        print(f"OOM Errors: {result.oom_errors}")
        print(f"Success Rate: {result.successful_runs/result.num_runs*100:.1f}%")
        
        if result.avg_time > 0:
            print(f"\nTiming:")
            print(f"  Average: {result.avg_time:.2f}s")
            print(f"  Min: {result.min_time:.2f}s")
            print(f"  Max: {result.max_time:.2f}s")
            print(f"  Std Dev: {result.std_time:.2f}s")
            
            print(f"\nPhase Breakdown:")
            print(f"  Encoding: {result.avg_encoding_time:.2f}s ({result.avg_encoding_time/result.avg_time*100:.1f}%)")
            print(f"  Diffusion: {result.avg_diffusion_time:.2f}s ({result.avg_diffusion_time/result.avg_time*100:.1f}%)")
            print(f"  Decoding: {result.avg_decoding_time:.2f}s ({result.avg_decoding_time/result.avg_time*100:.1f}%)")
            
            print(f"\nMemory:")
            print(f"  Average VRAM: {result.avg_vram_gb:.2f}GB")
            print(f"  Peak VRAM: {result.peak_vram_gb:.2f}GB")
    
    def compare_configs(self,
                       prompt: str,
                       configs: List[Dict],
                       num_runs: int = 3) -> List[BenchmarkResult]:
        """
        Compare multiple configurations
        
        Args:
            prompt: Test prompt
            configs: List of generation parameter configs
            num_runs: Number of runs per config
        
        Returns:
            List of BenchmarkResults
        """
        results = []
        
        for i, config in enumerate(configs):
            logger.info(f"\n{'='*70}")
            logger.info(f"Testing config {i+1}/{len(configs)}")
            logger.info(f"{'='*70}")
            
            result = self.benchmark_config(prompt, num_runs, **config)
            results.append(result)
            
            # Small delay between configs
            time.sleep(2)
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: List[BenchmarkResult]):
        """Print comparison of results"""
        print("\n" + "=" * 70)
        print("CONFIGURATION COMPARISON")
        print("=" * 70)
        
        print(f"\n{'Config':<20} {'Avg Time':<12} {'Peak VRAM':<12} {'Success Rate':<15}")
        print("-" * 70)
        
        for i, result in enumerate(results):
            config_str = f"Config {i+1}"
            avg_time = f"{result.avg_time:.2f}s"
            peak_vram = f"{result.peak_vram_gb:.2f}GB"
            success_rate = f"{result.successful_runs/result.num_runs*100:.1f}%"
            
            print(f"{config_str:<20} {avg_time:<12} {peak_vram:<12} {success_rate:<15}")


class DebugLogger:
    """
    Enhanced logging untuk debugging
    """
    
    def __init__(self, log_dir: str = "debug_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_execution(self, job_id: str, context: Dict, result: Dict):
        """Log execution details"""
        log_data = {
            "job_id": job_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "result": result
        }
        
        log_file = self.log_dir / f"{self.session_id}_{job_id}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Execution log saved to {log_file}")
    
    def log_error(self, job_id: str, error: Exception, traceback: str):
        """Log error details"""
        error_data = {
            "job_id": job_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback
        }
        
        error_file = self.log_dir / f"{self.session_id}_error_{job_id}.json"
        with open(error_file, 'w') as f:
            json.dump(error_data, f, indent=2)
        
        logger.error(f"Error log saved to {error_file}")
    
    def get_session_logs(self) -> List[Path]:
        """Get all logs for current session"""
        return list(self.log_dir.glob(f"{self.session_id}_*.json"))


def profile_memory_usage(func):
    """
    Decorator untuk profiling memory usage
    """
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return func(*args, **kwargs)
        
        # Clear cache before
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Record start
        start_allocated = torch.cuda.memory_allocated() / 1e9
        
        # Run function
        result = func(*args, **kwargs)
        
        # Record end
        end_allocated = torch.cuda.memory_allocated() / 1e9
        peak_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        # Log
        logger.info(f"Memory Profile for {func.__name__}:")
        logger.info(f"  Start: {start_allocated:.2f}GB")
        logger.info(f"  End: {end_allocated:.2f}GB")
        logger.info(f"  Peak: {peak_allocated:.2f}GB")
        logger.info(f"  Delta: {end_allocated - start_allocated:.2f}GB")
        
        return result
    
    return wrapper


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Utilities Demo")
    print("=" * 70)
    
    if torch.cuda.is_available():
        # Demo memory monitor
        print("\n1. Memory Monitor Demo")
        monitor = MemoryMonitor()
        monitor.start()
        
        # Simulate some work
        print("Monitoring memory for 3 seconds...")
        time.sleep(3)
        
        stats = monitor.stop()
        print(f"Collected {stats['num_samples']} samples")
        print(f"Peak VRAM: {stats['peak_allocated_gb']:.2f}GB")
        
        # Demo debug logger
        print("\n2. Debug Logger Demo")
        debug_logger = DebugLogger()
        
        # Log sample execution
        debug_logger.log_execution(
            job_id="test-job-001",
            context={"prompt": "test", "steps": 50},
            result={"status": "completed", "time": 15.5}
        )
        
        print(f"Session logs: {len(debug_logger.get_session_logs())} files")
        
        # Demo profiling decorator
        print("\n3. Memory Profiling Demo")
        
        @profile_memory_usage
        def allocate_tensor():
            tensor = torch.randn(1000, 1000, device="cuda")
            return tensor
        
        result = allocate_tensor()
        del result
        
    else:
        print("CUDA not available, skipping demos")
