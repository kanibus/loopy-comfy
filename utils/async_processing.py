# -*- coding: utf-8 -*-
"""
Async Processing Queues for LoopyComfy v2.0
Implements asyncio-based queues for high-throughput video processing.
"""

import asyncio
import time
import logging
from asyncio import Queue, Event, Semaphore
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Callable, Tuple, AsyncGenerator
import threading

import numpy as np

from .performance_optimizations import default_performance_monitor
from .security_utils import InputValidator, SecurityError

# Configure async logger
async_logger = logging.getLogger('loopycomfy.async')


class AsyncFrameQueue:
    """
    High-performance async frame processing queue.
    Provides non-blocking frame processing with backpressure control.
    """
    
    def __init__(self, max_size: int = 1000, max_workers: int = 4):
        """
        Initialize async frame queue.
        
        Args:
            max_size: Maximum queue size for backpressure
            max_workers: Maximum number of worker threads
        """
        self.max_size = max_size
        self.max_workers = max_workers
        
        # Async components
        self.input_queue = Queue(maxsize=max_size)
        self.output_queue = Queue(maxsize=max_size)
        self.shutdown_event = Event()
        
        # Processing semaphore for worker control
        self.worker_semaphore = Semaphore(max_workers)
        
        # Statistics
        self.stats = {
            'frames_queued': 0,
            'frames_processed': 0,
            'frames_dropped': 0,
            'processing_errors': 0,
            'queue_full_events': 0
        }
        self.stats_lock = threading.Lock()
        
        # Worker tasks
        self.worker_tasks = []
        self.is_running = False
        
        async_logger.info(f"AsyncFrameQueue initialized: max_size={max_size}, workers={max_workers}")
    
    async def start(self) -> None:
        """Start async processing workers."""
        if self.is_running:
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start worker tasks
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker_loop(i))
            self.worker_tasks.append(task)
        
        async_logger.info(f"AsyncFrameQueue started with {self.max_workers} workers")
    
    async def stop(self) -> None:
        """Stop async processing and cleanup."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        async_logger.info("AsyncFrameQueue stopped")
    
    async def queue_frame(self, frame: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Queue frame for processing.
        
        Args:
            frame: Frame to process
            metadata: Optional frame metadata
            
        Returns:
            True if queued successfully, False if queue full
        """
        try:
            frame_data = {
                'frame': frame,
                'metadata': metadata or {},
                'timestamp': time.time(),
                'frame_id': self.stats['frames_queued']
            }
            
            # Non-blocking queue put with timeout
            await asyncio.wait_for(self.input_queue.put(frame_data), timeout=0.1)
            
            with self.stats_lock:
                self.stats['frames_queued'] += 1
            
            return True
            
        except asyncio.TimeoutError:
            # Queue is full - implement backpressure
            with self.stats_lock:
                self.stats['queue_full_events'] += 1
                self.stats['frames_dropped'] += 1
            
            async_logger.warning("Frame dropped - queue full")
            return False
    
    async def get_processed_frame(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get processed frame from output queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Processed frame data or None if timeout
        """
        try:
            frame_data = await asyncio.wait_for(self.output_queue.get(), timeout=timeout)
            return frame_data
        except asyncio.TimeoutError:
            return None
    
    async def _worker_loop(self, worker_id: int) -> None:
        """
        Worker loop for processing frames.
        
        Args:
            worker_id: ID of this worker
        """
        async_logger.debug(f"Worker {worker_id} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get frame from input queue
                frame_data = await asyncio.wait_for(
                    self.input_queue.get(), timeout=0.5
                )
                
                # Process frame with semaphore control
                async with self.worker_semaphore:
                    processed_data = await self._process_frame_async(frame_data, worker_id)
                
                # Put processed frame to output queue
                await self.output_queue.put(processed_data)
                
                with self.stats_lock:
                    self.stats['frames_processed'] += 1
                
            except asyncio.TimeoutError:
                # No frames to process - continue
                continue
            except Exception as e:
                async_logger.error(f"Worker {worker_id} error: {e}")
                with self.stats_lock:
                    self.stats['processing_errors'] += 1
        
        async_logger.debug(f"Worker {worker_id} stopped")
    
    async def _process_frame_async(self, frame_data: Dict[str, Any], worker_id: int) -> Dict[str, Any]:
        """
        Process frame asynchronously.
        
        Args:
            frame_data: Frame data to process
            worker_id: ID of processing worker
            
        Returns:
            Processed frame data
        """
        start_time = time.time()
        
        try:
            frame = frame_data['frame']
            metadata = frame_data['metadata']
            
            # Run CPU-intensive processing in thread pool
            loop = asyncio.get_event_loop()
            processed_frame = await loop.run_in_executor(
                None, self._process_frame_sync, frame, metadata
            )
            
            processing_time = time.time() - start_time
            
            return {
                'frame': processed_frame,
                'metadata': metadata,
                'processing_time': processing_time,
                'worker_id': worker_id,
                'frame_id': frame_data['frame_id'],
                'timestamp': frame_data['timestamp']
            }
            
        except Exception as e:
            async_logger.error(f"Frame processing error: {e}")
            return {
                'frame': frame_data['frame'],  # Return original on error
                'metadata': frame_data['metadata'],
                'error': str(e),
                'processing_time': time.time() - start_time,
                'worker_id': worker_id,
                'frame_id': frame_data['frame_id'],
                'timestamp': frame_data['timestamp']
            }
    
    def _process_frame_sync(self, frame: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Synchronous frame processing (runs in thread pool).
        Override this method for custom processing.
        
        Args:
            frame: Input frame
            metadata: Frame metadata
            
        Returns:
            Processed frame
        """
        # Default: return frame unchanged
        return frame.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue processing statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Calculate additional metrics
        stats['queue_utilization'] = (
            self.input_queue.qsize() / self.max_size * 100
        )
        stats['processing_rate'] = (
            stats['frames_processed'] / max(1, time.time() - (getattr(self, '_start_time', time.time())))
        )
        
        return stats


class AsyncVideoPipeline:
    """
    Complete async video processing pipeline with stages.
    Provides high-throughput video processing with configurable stages.
    """
    
    def __init__(self, stages: List[Callable] = None, max_concurrent: int = 10):
        """
        Initialize async video pipeline.
        
        Args:
            stages: List of processing stage functions
            max_concurrent: Maximum concurrent video processing
        """
        self.stages = stages or []
        self.max_concurrent = max_concurrent
        self.processing_semaphore = Semaphore(max_concurrent)
        
        # Pipeline queues
        self.input_queue = Queue()
        self.output_queue = Queue()
        
        # Pipeline control
        self.is_running = False
        self.pipeline_tasks = []
        
        async_logger.info(f"AsyncVideoPipeline initialized with {len(self.stages)} stages")
    
    async def add_stage(self, stage_func: Callable) -> None:
        """Add processing stage to pipeline."""
        self.stages.append(stage_func)
        async_logger.info(f"Added stage to pipeline: {stage_func.__name__}")
    
    async def start(self) -> None:
        """Start async pipeline processing."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start pipeline workers
        for i in range(self.max_concurrent):
            task = asyncio.create_task(self._pipeline_worker(i))
            self.pipeline_tasks.append(task)
        
        async_logger.info("AsyncVideoPipeline started")
    
    async def stop(self) -> None:
        """Stop async pipeline processing."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel pipeline tasks
        for task in self.pipeline_tasks:
            task.cancel()
        
        if self.pipeline_tasks:
            await asyncio.gather(*self.pipeline_tasks, return_exceptions=True)
        
        self.pipeline_tasks.clear()
        async_logger.info("AsyncVideoPipeline stopped")
    
    async def process_video(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process video through pipeline.
        
        Args:
            video_data: Video data to process
            
        Returns:
            Processed video data
        """
        await self.input_queue.put(video_data)
        result = await self.output_queue.get()
        return result
    
    async def process_videos_batch(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process batch of videos through pipeline.
        
        Args:
            videos: List of video data
            
        Returns:
            List of processed video data
        """
        # Queue all videos
        tasks = []
        for video_data in videos:
            task = asyncio.create_task(self.process_video(video_data))
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        async_logger.info(f"Processed {len(successful_results)}/{len(videos)} videos successfully")
        return successful_results
    
    async def _pipeline_worker(self, worker_id: int) -> None:
        """
        Pipeline worker for processing videos.
        
        Args:
            worker_id: Worker identifier
        """
        while self.is_running:
            try:
                # Get video from input queue
                video_data = await asyncio.wait_for(
                    self.input_queue.get(), timeout=0.5
                )
                
                # Process through all stages with semaphore control
                async with self.processing_semaphore:
                    processed_data = await self._process_through_stages(video_data, worker_id)
                
                # Put result to output queue
                await self.output_queue.put(processed_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                async_logger.error(f"Pipeline worker {worker_id} error: {e}")
    
    async def _process_through_stages(self, video_data: Dict[str, Any], worker_id: int) -> Dict[str, Any]:
        """
        Process video data through all pipeline stages.
        
        Args:
            video_data: Video data to process
            worker_id: Processing worker ID
            
        Returns:
            Processed video data
        """
        current_data = video_data.copy()
        current_data['worker_id'] = worker_id
        current_data['processing_start'] = time.time()
        
        for i, stage in enumerate(self.stages):
            try:
                stage_start = time.time()
                
                # Run stage in thread pool for CPU-intensive work
                loop = asyncio.get_event_loop()
                current_data = await loop.run_in_executor(None, stage, current_data)
                
                stage_time = time.time() - stage_start
                current_data[f'stage_{i}_time'] = stage_time
                
            except Exception as e:
                async_logger.error(f"Stage {i} error: {e}")
                current_data[f'stage_{i}_error'] = str(e)
        
        current_data['total_processing_time'] = time.time() - current_data['processing_start']
        return current_data


class AsyncPerformanceMonitor:
    """
    Async performance monitoring for real-time optimization.
    """
    
    def __init__(self, monitoring_interval: float = 1.0):
        """
        Initialize async performance monitor.
        
        Args:
            monitoring_interval: Monitoring interval in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.metrics = {}
        self.is_monitoring = False
        self.monitor_task = None
        
    async def start_monitoring(self) -> None:
        """Start async performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        async_logger.info("Async performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop async performance monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        async_logger.info("Async performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect performance metrics
                await self._collect_metrics()
                
                # Wait for next monitoring interval
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                async_logger.error(f"Monitoring error: {e}")
    
    async def _collect_metrics(self) -> None:
        """Collect performance metrics asynchronously."""
        # Placeholder for metrics collection
        # Could monitor CPU, memory, queue sizes, etc.
        current_time = time.time()
        
        # Update metrics
        self.metrics.update({
            'timestamp': current_time,
            'monitoring_active': self.is_monitoring
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()


# Global async components
default_frame_queue = AsyncFrameQueue()
default_video_pipeline = AsyncVideoPipeline()
default_async_monitor = AsyncPerformanceMonitor()


async def setup_async_processing() -> Dict[str, Any]:
    """
    Setup and start all async processing components.
    
    Returns:
        Dictionary of async components
    """
    await default_frame_queue.start()
    await default_video_pipeline.start()
    await default_async_monitor.start_monitoring()
    
    async_logger.info("Async processing setup complete")
    
    return {
        'frame_queue': default_frame_queue,
        'video_pipeline': default_video_pipeline,
        'performance_monitor': default_async_monitor
    }


async def shutdown_async_processing() -> None:
    """Shutdown all async processing components."""
    await default_frame_queue.stop()
    await default_video_pipeline.stop()
    await default_async_monitor.stop_monitoring()
    
    async_logger.info("Async processing shutdown complete")