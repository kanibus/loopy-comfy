# -*- coding: utf-8 -*-
"""
Memory Management Utilities for LoopyComfy
Provides memory-bounded processing, cleanup, and monitoring.
"""

import os
import gc
import psutil
import tracemalloc
import weakref
import logging
from contextlib import contextmanager
from collections import deque
from typing import List, Optional, Any, Generator, Dict
import numpy as np

# Configure logger
memory_logger = logging.getLogger('loopycomfy.memory')


class MemoryBoundedError(Exception):
    """Raised when memory limits are exceeded."""
    pass


class MemoryMonitor:
    """Monitor and enforce memory usage limits."""
    
    def __init__(self, max_memory_bytes: int = 8 * 1024 * 1024 * 1024):
        """
        Initialize memory monitor.
        
        Args:
            max_memory_bytes: Maximum memory usage in bytes (default: 8GB)
        """
        self.max_memory_bytes = max_memory_bytes
        self.initial_memory = 0
        self.peak_memory = 0
        self._tracemalloc_started = False
    
    def __enter__(self):
        """Start memory monitoring."""
        try:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self._tracemalloc_started = True
            self.initial_memory = tracemalloc.get_traced_memory()[0]
        except Exception as e:
            memory_logger.warning(f"Failed to start tracemalloc: {e}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop memory monitoring."""
        if self._tracemalloc_started:
            try:
                self.peak_memory = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
            except Exception as e:
                memory_logger.warning(f"Failed to stop tracemalloc: {e}")
    
    @property
    def current_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            if tracemalloc.is_tracing():
                return tracemalloc.get_traced_memory()[0] - self.initial_memory
            else:
                # Fallback to psutil
                return psutil.Process().memory_info().rss
        except Exception:
            return 0
    
    @property
    def current_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.current_usage / 1024 / 1024
    
    def check_limits(self, raise_on_exceed: bool = True) -> bool:
        """
        Check if memory usage exceeds limits.
        
        Args:
            raise_on_exceed: Whether to raise exception on limit exceed
            
        Returns:
            True if within limits, False otherwise
            
        Raises:
            MemoryBoundedError: If limits exceeded and raise_on_exceed=True
        """
        current = self.current_usage
        if current > self.max_memory_bytes:
            message = f"Memory limit exceeded: {current/1024/1024:.1f}MB > {self.max_memory_bytes/1024/1024:.1f}MB"
            memory_logger.error(message)
            
            if raise_on_exceed:
                raise MemoryBoundedError(message)
            return False
        return True
    
    def should_flush(self, threshold_ratio: float = 0.8) -> bool:
        """Check if memory usage requires flushing."""
        return self.current_usage > (self.max_memory_bytes * threshold_ratio)


class BoundedFrameBuffer:
    """Memory-bounded frame buffer with automatic cleanup."""
    
    def __init__(self, max_frames: int = 1000, max_memory_mb: int = 4000):
        """
        Initialize bounded frame buffer.
        
        Args:
            max_frames: Maximum number of frames to hold in memory
            max_memory_mb: Maximum memory usage for frames
        """
        self.max_frames = max_frames
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.frames = deque(maxlen=max_frames)
        self.current_memory = 0
        self._temp_files = []
        self._frame_refs = weakref.WeakSet()
    
    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add frame to buffer with memory management.
        
        Args:
            frame: Frame to add
        """
        frame_size = frame.nbytes
        
        # Check if adding this frame would exceed memory limits
        if self.current_memory + frame_size > self.max_memory_bytes:
            self._flush_to_disk()
        
        self.frames.append(frame.copy())
        self.current_memory += frame_size
        self._frame_refs.add(frame)
    
    def extend(self, frames: List[np.ndarray]) -> None:
        """Add multiple frames with memory management."""
        for frame in frames:
            self.add_frame(frame)
    
    def get_all_frames(self) -> List[np.ndarray]:
        """Get all frames, loading from disk if necessary."""
        all_frames = list(self.frames)
        
        # Load any frames that were flushed to disk
        for temp_file in self._temp_files:
            if os.path.exists(temp_file):
                try:
                    disk_frames = np.load(temp_file)
                    all_frames.extend([frame for frame in disk_frames])
                    os.unlink(temp_file)  # Clean up temp file
                except Exception as e:
                    memory_logger.error(f"Failed to load frames from {temp_file}: {e}")
        
        self._temp_files.clear()
        return all_frames
    
    def _flush_to_disk(self) -> None:
        """Flush oldest frames to disk to free memory."""
        if not self.frames:
            return
        
        try:
            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), "temp", "loopy_comfy_frames")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save half the frames to disk
            flush_count = len(self.frames) // 2
            frames_to_flush = []
            
            for _ in range(flush_count):
                if self.frames:
                    frame = self.frames.popleft()
                    frames_to_flush.append(frame)
                    self.current_memory -= frame.nbytes
            
            if frames_to_flush:
                temp_file = os.path.join(temp_dir, f"frames_{len(self._temp_files)}.npy")
                np.save(temp_file, np.array(frames_to_flush))
                self._temp_files.append(temp_file)
                
                memory_logger.info(f"Flushed {len(frames_to_flush)} frames to {temp_file}")
                
                # Force garbage collection
                del frames_to_flush
                gc.collect()
        
        except Exception as e:
            memory_logger.error(f"Failed to flush frames to disk: {e}")
    
    def clear(self) -> None:
        """Clear all frames and cleanup temp files."""
        self.frames.clear()
        self.current_memory = 0
        
        # Clean up temp files
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                memory_logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        self._temp_files.clear()
        
        # Force garbage collection
        gc.collect()
    
    def __del__(self):
        """Cleanup on destruction."""
        self.clear()


@contextmanager
def managed_frame_processing(max_memory_mb: int = 7000, 
                           max_frames: int = 1000) -> Generator[BoundedFrameBuffer, None, None]:
    """
    Context manager for memory-bounded frame processing.
    
    Args:
        max_memory_mb: Maximum memory usage in MB
        max_frames: Maximum frames in memory buffer
        
    Yields:
        BoundedFrameBuffer for managing frames
    """
    monitor = MemoryMonitor(max_memory_mb * 1024 * 1024)
    buffer = BoundedFrameBuffer(max_frames, max_memory_mb // 2)  # Use half memory for buffer
    
    try:
        with monitor:
            yield buffer
            
            # Final memory check
            monitor.check_limits(raise_on_exceed=True)
            
    finally:
        buffer.clear()
        gc.collect()


class VideoResourceManager:
    """Manage video resources with automatic cleanup."""
    
    def __init__(self):
        self._video_captures = weakref.WeakSet()
        self._temp_files = []
    
    @contextmanager
    def video_capture(self, video_path: str):
        """Context manager for video capture with automatic cleanup."""
        import cv2
        
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            
            self._video_captures.add(cap)
            yield cap
            
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception as e:
                    memory_logger.warning(f"Failed to release video capture: {e}")
    
    def add_temp_file(self, file_path: str) -> None:
        """Add a temporary file for cleanup."""
        self._temp_files.append(file_path)
    
    def cleanup_all(self) -> None:
        """Cleanup all managed resources."""
        # Cleanup video captures
        for cap in list(self._video_captures):
            try:
                if hasattr(cap, 'release'):
                    cap.release()
            except Exception as e:
                memory_logger.warning(f"Failed to cleanup video capture: {e}")
        
        # Cleanup temp files
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                memory_logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        self._temp_files.clear()
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup_all()


# Global instance for easy access
default_resource_manager = VideoResourceManager()


def force_cleanup() -> None:
    """Force cleanup of all resources and garbage collection."""
    try:
        # Cleanup default resource manager
        default_resource_manager.cleanup_all()
        
        # Force garbage collection multiple times
        for _ in range(3):
            collected = gc.collect()
            if collected == 0:
                break
        
        memory_logger.info("Forced cleanup completed")
        
    except Exception as e:
        memory_logger.error(f"Error during force cleanup: {e}")


def get_memory_info() -> Dict[str, Any]:
    """Get current memory information."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'total_mb': psutil.virtual_memory().total / 1024 / 1024
        }
    except Exception as e:
        memory_logger.error(f"Failed to get memory info: {e}")
        return {}