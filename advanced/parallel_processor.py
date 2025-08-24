# -*- coding: utf-8 -*-
"""
Parallel CPU Processor for LoopyComfy v2.0

This module provides multi-threaded CPU processing with intelligent
load balancing and automatic worker scaling.
"""

import numpy as np
import time
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Tuple
import warnings


class WorkerStats:
    """Statistics tracking for individual workers."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.tasks_completed = 0
        self.total_processing_time = 0.0
        self.last_task_time = 0.0
        self.error_count = 0
        self.start_time = time.time()
    
    def update(self, processing_time: float):
        """Update worker statistics."""
        self.tasks_completed += 1
        self.total_processing_time += processing_time
        self.last_task_time = processing_time
    
    def record_error(self):
        """Record an error for this worker."""
        self.error_count += 1
    
    def get_average_time(self) -> float:
        """Get average processing time."""
        if self.tasks_completed == 0:
            return 0.0
        return self.total_processing_time / self.tasks_completed
    
    def get_throughput(self) -> float:
        """Get tasks per second."""
        uptime = time.time() - self.start_time
        if uptime <= 0:
            return 0.0
        return self.tasks_completed / uptime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'worker_id': self.worker_id,
            'tasks_completed': self.tasks_completed,
            'total_processing_time': self.total_processing_time,
            'average_time': self.get_average_time(),
            'last_task_time': self.last_task_time,
            'error_count': self.error_count,
            'throughput': self.get_throughput(),
            'uptime': time.time() - self.start_time
        }


def _process_frame_worker(frame_data: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray, float]:
    """Worker function for processing individual frames."""
    worker_id, frame = frame_data
    start_time = time.time()
    
    try:
        # Frame processing logic
        processed_frame = _apply_frame_processing(frame)
        processing_time = time.time() - start_time
        return (worker_id, processed_frame, processing_time)
        
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        processing_time = time.time() - start_time
        return (worker_id, frame, processing_time)  # Return original on error


def _apply_frame_processing(frame: np.ndarray) -> np.ndarray:
    """Apply frame processing operations."""
    try:
        import cv2
        
        # Basic processing pipeline
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convert BGR to RGB
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply subtle enhancement (optional)
            processed = cv2.convertScaleAbs(processed, alpha=1.05, beta=2)
            
            return processed
        else:
            return frame
            
    except ImportError:
        # Fallback without OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return frame[:, :, [2, 1, 0]]  # Simple BGR to RGB
        return frame


class ParallelCPUProcessor:
    """Multi-threaded CPU processor with load balancing."""
    
    def __init__(self, num_workers: Optional[int] = None, use_processes: bool = False):
        """Initialize parallel processor."""
        # Worker configuration
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overhead
        
        self.num_workers = num_workers
        self.use_processes = use_processes  # ProcessPool vs ThreadPool
        
        # Executor
        self.executor = None
        self.executor_type = "ProcessPoolExecutor" if use_processes else "ThreadPoolExecutor"
        
        # Statistics
        self.worker_stats = {i: WorkerStats(i) for i in range(num_workers)}
        self.global_stats = {
            'total_frames_processed': 0,
            'total_processing_time': 0.0,
            'total_batches': 0,
            'error_count': 0,
            'start_time': time.time()
        }
        
        # Performance tracking
        self.processing_times = []
        self.throughput_history = []
        
        # Thread safety
        self.stats_lock = threading.Lock()
        
        self._initialize_executor()
        
        print(f"Parallel CPU Processor initialized: {num_workers} workers ({self.executor_type})")
    
    def _initialize_executor(self):
        """Initialize the thread/process pool executor."""
        try:
            if self.use_processes:
                self.executor = ProcessPoolExecutor(
                    max_workers=self.num_workers,
                    mp_context=mp.get_context('spawn')  # More reliable on Windows
                )
            else:
                self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
                
        except Exception as e:
            print(f"Failed to initialize {self.executor_type}: {e}")
            print("Falling back to ThreadPoolExecutor")
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
            self.executor_type = "ThreadPoolExecutor (fallback)"
    
    def process_frame_batch(self, frames: np.ndarray) -> np.ndarray:
        """Process batch of frames in parallel."""
        if frames.shape[0] == 0:
            return frames
        
        batch_start_time = time.time()
        
        try:
            # Prepare work items
            work_items = [(i % self.num_workers, frame) for i, frame in enumerate(frames)]
            
            # Submit work to executor
            future_to_index = {}
            for i, work_item in enumerate(work_items):
                future = self.executor.submit(_process_frame_worker, work_item)
                future_to_index[future] = i
            
            # Collect results
            results = [None] * len(frames)
            worker_times = []
            
            for future in as_completed(future_to_index.keys()):
                index = future_to_index[future]
                
                try:
                    worker_id, processed_frame, processing_time = future.result()
                    results[index] = processed_frame
                    worker_times.append(processing_time)
                    
                    # Update worker stats
                    with self.stats_lock:
                        self.worker_stats[worker_id].update(processing_time)
                        
                except Exception as e:
                    print(f"Frame processing failed: {e}")
                    results[index] = frames[index]  # Use original frame
                    
                    with self.stats_lock:
                        worker_id = work_items[index][0]
                        self.worker_stats[worker_id].record_error()
                        self.global_stats['error_count'] += 1
            
            # Update global statistics
            batch_time = time.time() - batch_start_time
            
            with self.stats_lock:
                self.global_stats['total_frames_processed'] += len(frames)
                self.global_stats['total_processing_time'] += batch_time
                self.global_stats['total_batches'] += 1
                
                # Track performance
                self.processing_times.append(batch_time)
                if len(self.processing_times) > 100:
                    self.processing_times = self.processing_times[-100:]
                
                throughput = len(frames) / batch_time
                self.throughput_history.append(throughput)
                if len(self.throughput_history) > 100:
                    self.throughput_history = self.throughput_history[-100:]
            
            return np.array(results)
            
        except Exception as e:
            print(f"Batch processing failed: {e}")
            with self.stats_lock:
                self.global_stats['error_count'] += 1
            
            # Fallback to sequential processing
            return self._process_batch_sequential(frames)
    
    def _process_batch_sequential(self, frames: np.ndarray) -> np.ndarray:
        """Fallback sequential processing."""
        print("Falling back to sequential processing")
        
        processed_frames = []
        for frame in frames:
            processed_frame = _apply_frame_processing(frame)
            processed_frames.append(processed_frame)
        
        return np.array(processed_frames)
    
    def process_frame_batch_adaptive(self, frames: np.ndarray, target_latency: float = 100.0) -> np.ndarray:
        """Process frames with adaptive batching based on target latency."""
        if frames.shape[0] == 0:
            return frames
        
        # Estimate processing time based on history
        avg_time_per_frame = self._estimate_frame_processing_time()
        
        if avg_time_per_frame > 0:
            # Calculate optimal batch size for target latency
            optimal_batch_size = max(1, int(target_latency / (avg_time_per_frame * 1000)))
            optimal_batch_size = min(optimal_batch_size, frames.shape[0])
        else:
            optimal_batch_size = min(self.num_workers * 2, frames.shape[0])
        
        # Process in optimally sized batches
        if optimal_batch_size >= frames.shape[0]:
            return self.process_frame_batch(frames)
        else:
            results = []
            for i in range(0, frames.shape[0], optimal_batch_size):
                batch = frames[i:i + optimal_batch_size]
                batch_result = self.process_frame_batch(batch)
                results.append(batch_result)
            
            return np.concatenate(results, axis=0)
    
    def _estimate_frame_processing_time(self) -> float:
        """Estimate average processing time per frame."""
        if not self.processing_times:
            return 0.0
        
        # Get recent average batch time
        recent_times = self.processing_times[-20:]  # Last 20 batches
        avg_batch_time = np.mean(recent_times)
        
        # Estimate frames per batch (approximate)
        if self.global_stats['total_batches'] > 0:
            avg_frames_per_batch = self.global_stats['total_frames_processed'] / self.global_stats['total_batches']
            return avg_batch_time / max(1, avg_frames_per_batch)
        
        return 0.0
    
    def resize_frames_parallel(self, frames: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frames in parallel."""
        if frames.shape[0] == 0:
            return frames
        
        def resize_worker(frame_data):
            worker_id, frame = frame_data
            start_time = time.time()
            
            try:
                import cv2
                resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                processing_time = time.time() - start_time
                return (worker_id, resized, processing_time)
            except ImportError:
                # Fallback without OpenCV (simple nearest neighbor)
                height, width = frame.shape[:2]
                out_width, out_height = target_size
                
                scale_x = width / out_width
                scale_y = height / out_height
                
                if len(frame.shape) == 3:
                    resized = np.zeros((out_height, out_width, frame.shape[2]), dtype=frame.dtype)
                    for y in range(out_height):
                        for x in range(out_width):
                            src_x = min(int(x * scale_x), width - 1)
                            src_y = min(int(y * scale_y), height - 1)
                            resized[y, x] = frame[src_y, src_x]
                else:
                    resized = np.zeros((out_height, out_width), dtype=frame.dtype)
                    for y in range(out_height):
                        for x in range(out_width):
                            src_x = min(int(x * scale_x), width - 1)
                            src_y = min(int(y * scale_y), height - 1)
                            resized[y, x] = frame[src_y, src_x]
                
                processing_time = time.time() - start_time
                return (worker_id, resized, processing_time)
        
        # Prepare work items
        work_items = [(i % self.num_workers, frame) for i, frame in enumerate(frames)]
        
        try:
            # Submit resize tasks
            future_to_index = {}
            for i, work_item in enumerate(work_items):
                future = self.executor.submit(resize_worker, work_item)
                future_to_index[future] = i
            
            # Collect results
            results = [None] * len(frames)
            
            for future in as_completed(future_to_index.keys()):
                index = future_to_index[future]
                
                try:
                    worker_id, resized_frame, processing_time = future.result()
                    results[index] = resized_frame
                    
                    with self.stats_lock:
                        self.worker_stats[worker_id].update(processing_time)
                        
                except Exception as e:
                    print(f"Resize failed for frame {index}: {e}")
                    results[index] = frames[index]  # Keep original size
            
            return np.array(results)
            
        except Exception as e:
            print(f"Parallel resize failed: {e}")
            return frames  # Return original frames
    
    def get_worker_load_balance(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self.stats_lock:
            worker_loads = []
            for worker_stat in self.worker_stats.values():
                worker_loads.append({
                    'worker_id': worker_stat.worker_id,
                    'tasks_completed': worker_stat.tasks_completed,
                    'average_time': worker_stat.get_average_time(),
                    'throughput': worker_stat.get_throughput(),
                    'error_rate': worker_stat.error_count / max(1, worker_stat.tasks_completed)
                })
            
            # Calculate load distribution metrics
            task_counts = [w['tasks_completed'] for w in worker_loads]
            throughputs = [w['throughput'] for w in worker_loads]
            
            load_balance_score = 1.0
            if len(task_counts) > 1 and max(task_counts) > 0:
                load_balance_score = min(task_counts) / max(task_counts)
            
            return {
                'workers': worker_loads,
                'load_balance_score': load_balance_score,
                'total_tasks': sum(task_counts),
                'average_throughput': np.mean(throughputs) if throughputs else 0.0,
                'throughput_variance': np.var(throughputs) if throughputs else 0.0
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self.stats_lock:
            uptime = time.time() - self.global_stats['start_time']
            
            stats = {
                'num_workers': self.num_workers,
                'executor_type': self.executor_type,
                'uptime_seconds': uptime,
                'total_frames_processed': self.global_stats['total_frames_processed'],
                'total_batches': self.global_stats['total_batches'],
                'total_errors': self.global_stats['error_count'],
                'average_batch_time': (
                    np.mean(self.processing_times) if self.processing_times else 0.0
                ),
                'average_throughput_fps': (
                    np.mean(self.throughput_history) if self.throughput_history else 0.0
                ),
                'peak_throughput_fps': (
                    np.max(self.throughput_history) if self.throughput_history else 0.0
                ),
                'overall_fps': (
                    self.global_stats['total_frames_processed'] / uptime if uptime > 0 else 0.0
                ),
                'error_rate': (
                    self.global_stats['error_count'] / 
                    max(1, self.global_stats['total_frames_processed'])
                )
            }
            
            # Add worker-specific stats
            stats['worker_stats'] = [stat.to_dict() for stat in self.worker_stats.values()]
            stats['load_balance'] = self.get_worker_load_balance()
            
            return stats
    
    def benchmark_performance(self, num_frames: int = 200) -> Dict[str, float]:
        """Benchmark parallel processing performance."""
        print(f"Benchmarking parallel processor ({num_frames} frames, {self.num_workers} workers)...")
        
        # Generate test data
        test_frames = np.random.randint(0, 255, (num_frames, 256, 256, 3), dtype=np.uint8)
        
        # Clear statistics
        with self.stats_lock:
            for stat in self.worker_stats.values():
                stat.__init__(stat.worker_id)  # Reset stats
            
            self.global_stats = {
                'total_frames_processed': 0,
                'total_processing_time': 0.0,
                'total_batches': 0,
                'error_count': 0,
                'start_time': time.time()
            }
            self.processing_times.clear()
            self.throughput_history.clear()
        
        # Benchmark different batch sizes
        batch_sizes = [1, 5, 10, 20, 50]
        batch_results = {}
        
        for batch_size in batch_sizes:
            if batch_size > num_frames:
                continue
            
            print(f"  Testing batch size: {batch_size}")
            
            start_time = time.time()
            
            for i in range(0, num_frames, batch_size):
                end_idx = min(i + batch_size, num_frames)
                batch = test_frames[i:end_idx]
                _ = self.process_frame_batch(batch)
            
            elapsed_time = time.time() - start_time
            fps = num_frames / elapsed_time
            
            batch_results[f'batch_size_{batch_size}_fps'] = fps
            batch_results[f'batch_size_{batch_size}_time'] = elapsed_time
        
        # Test adaptive processing
        start_time = time.time()
        _ = self.process_frame_batch_adaptive(test_frames, target_latency=50.0)
        adaptive_time = time.time() - start_time
        adaptive_fps = num_frames / adaptive_time
        
        results = {
            'num_workers': self.num_workers,
            'executor_type': self.executor_type,
            'adaptive_fps': adaptive_fps,
            'adaptive_time': adaptive_time,
            **batch_results
        }
        
        print(f"Parallel Benchmark Results:")
        print(f"  Workers: {self.num_workers}")
        print(f"  Best FPS: {max([v for k, v in results.items() if k.endswith('_fps')]):.1f}")
        print(f"  Adaptive FPS: {adaptive_fps:.1f}")
        
        return results
    
    def scale_workers(self, new_worker_count: int):
        """Dynamically scale the number of workers."""
        if new_worker_count == self.num_workers:
            return
        
        print(f"Scaling workers from {self.num_workers} to {new_worker_count}")
        
        # Shutdown current executor
        old_executor = self.executor
        old_executor.shutdown(wait=True)
        
        # Update worker count and create new executor
        self.num_workers = new_worker_count
        self.worker_stats = {i: WorkerStats(i) for i in range(new_worker_count)}
        
        self._initialize_executor()
        
        print(f"Worker scaling complete: {self.num_workers} workers active")
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up parallel processor...")
        
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        # Clear statistics
        with self.stats_lock:
            self.processing_times.clear()
            self.throughput_history.clear()
            self.worker_stats.clear()
        
        print("Parallel processor cleanup complete")
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass