# -*- coding: utf-8 -*-
"""
GPU Accelerated Processor for LoopyComfy v2.0

This module provides GPU acceleration with proper resource management,
CUDA optimization, and automatic fallback to CPU processing.
"""

import numpy as np
import time
import warnings
import threading
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple
import gc


class GPUAcceleratedProcessor:
    """GPU processing with proper resource management and fallbacks."""
    
    def __init__(self, config, resource_monitor):
        """Initialize GPU processor."""
        self.config = config
        self.monitor = resource_monitor
        
        # GPU state
        self.device = None
        self.cuda_available = False
        self.cupy_available = False
        self.pytorch_available = False
        
        # Memory management
        self.memory_pool = None
        self.memory_limit = None
        
        # Performance tracking
        self.processing_times = []
        self.gpu_utilization = []
        self.memory_usage = []
        
        # Kernels
        self.kernels = {}
        self.kernels_compiled = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize GPU resources
        self._initialize_gpu()
        
        print(f"GPU Processor initialized: CUDA={self.cuda_available}, CuPy={self.cupy_available}")
    
    def _initialize_gpu(self):
        """Initialize GPU resources with fallback detection."""
        # Check PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.cuda_available = True
                self.pytorch_available = True
                
                # Set memory pool limit
                mempool_size = self.config.get('gpu.memory_pool_size_mb', 'auto')
                if mempool_size == 'auto':
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    self.memory_limit = int(total_memory * 0.25)  # Use 25% of GPU memory
                else:
                    self.memory_limit = mempool_size * 1024 * 1024
                
                print(f"PyTorch CUDA initialized (limit: {self.memory_limit / (1024**2):.0f}MB)")
        except ImportError:
            print("PyTorch not available")
        
        # Check CuPy
        try:
            import cupy as cp
            if cp.cuda.is_available():
                self.cupy_available = True
                
                # Set up memory pool
                if self.memory_limit:
                    self.memory_pool = cp.get_default_memory_pool()
                    self.memory_pool.set_limit(size=self.memory_limit)
                
                print(f"CuPy initialized (limit: {self.memory_limit / (1024**2):.0f}MB)")
                
                # Compile kernels
                self._compile_kernels()
                
        except ImportError:
            print("CuPy not available")
        
        if not self.cuda_available:
            print("No CUDA support available - will use CPU fallback")
    
    def _compile_kernels(self):
        """Compile CUDA kernels for common operations."""
        try:
            import cupy as cp
            
            # BGR to RGB conversion kernel
            self.kernels['bgr_to_rgb'] = cp.RawKernel(r'''
            extern "C" __global__
            void bgr_to_rgb_kernel(unsigned char* input, unsigned char* output, 
                                   int width, int height, int channels) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int total_pixels = width * height;
                
                if (idx < total_pixels) {
                    int pixel_idx = idx * channels;
                    
                    // Swap B and R channels
                    output[pixel_idx] = input[pixel_idx + 2];      // R
                    output[pixel_idx + 1] = input[pixel_idx + 1];  // G (unchanged)
                    output[pixel_idx + 2] = input[pixel_idx];      // B
                    
                    // Handle alpha channel if present
                    if (channels == 4) {
                        output[pixel_idx + 3] = input[pixel_idx + 3];  // A
                    }
                }
            }
            ''', 'bgr_to_rgb_kernel')
            
            # Frame resizing kernel
            self.kernels['resize_bilinear'] = cp.RawKernel(r'''
            extern "C" __global__
            void resize_bilinear_kernel(unsigned char* input, unsigned char* output,
                                       int in_width, int in_height,
                                       int out_width, int out_height, int channels) {
                int out_x = blockIdx.x * blockDim.x + threadIdx.x;
                int out_y = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (out_x >= out_width || out_y >= out_height) return;
                
                float scale_x = (float)in_width / out_width;
                float scale_y = (float)in_height / out_height;
                
                float src_x = out_x * scale_x;
                float src_y = out_y * scale_y;
                
                int x1 = (int)src_x;
                int y1 = (int)src_y;
                int x2 = min(x1 + 1, in_width - 1);
                int y2 = min(y1 + 1, in_height - 1);
                
                float dx = src_x - x1;
                float dy = src_y - y1;
                
                for (int c = 0; c < channels; c++) {
                    float p11 = input[(y1 * in_width + x1) * channels + c];
                    float p12 = input[(y1 * in_width + x2) * channels + c];
                    float p21 = input[(y2 * in_width + x1) * channels + c];
                    float p22 = input[(y2 * in_width + x2) * channels + c];
                    
                    float p1 = p11 * (1 - dx) + p12 * dx;
                    float p2 = p21 * (1 - dx) + p22 * dx;
                    float result = p1 * (1 - dy) + p2 * dy;
                    
                    output[(out_y * out_width + out_x) * channels + c] = (unsigned char)result;
                }
            }
            ''', 'resize_bilinear_kernel')
            
            # Test kernels
            test_input = cp.random.randint(0, 255, (100, 100, 3), dtype=cp.uint8)
            test_output = cp.zeros_like(test_input)
            
            threads_per_block = (16, 16)
            blocks_per_grid = (
                (100 + threads_per_block[0] - 1) // threads_per_block[0],
                (100 + threads_per_block[1] - 1) // threads_per_block[1]
            )
            
            self.kernels['bgr_to_rgb'](
                blocks_per_grid, threads_per_block,
                (test_input, test_output, 100, 100, 3)
            )
            
            cp.cuda.Stream.null.synchronize()
            
            self.kernels_compiled = True
            print("CUDA kernels compiled successfully")
            
        except Exception as e:
            print(f"Failed to compile CUDA kernels: {e}")
            self.kernels_compiled = False
    
    @contextmanager
    def gpu_context(self):
        """Context manager for GPU operations with cleanup."""
        try:
            yield
        finally:
            self._cleanup_gpu_memory()
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory."""
        try:
            if self.pytorch_available:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            
            if self.cupy_available:
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
                if self.memory_pool:
                    self.memory_pool.free_all_blocks()
            
        except Exception as e:
            print(f"GPU cleanup error: {e}")
    
    def process_frame_gpu(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame on GPU with fallback."""
        if not self.cuda_available:
            return self._process_frame_cpu_fallback(frame)
        
        with self.lock:
            try:
                start_time = time.time()
                
                with self.gpu_context():
                    if self.cupy_available and self.kernels_compiled:
                        result = self._process_frame_cupy(frame)
                    elif self.pytorch_available:
                        result = self._process_frame_pytorch(frame)
                    else:
                        result = self._process_frame_cpu_fallback(frame)
                
                processing_time = (time.time() - start_time) * 1000
                self._update_performance_metrics(processing_time)
                
                return result
                
            except Exception as e:
                print(f"GPU frame processing failed: {e}")
                return self._process_frame_cpu_fallback(frame)
    
    def _process_frame_cupy(self, frame: np.ndarray) -> np.ndarray:
        """Process frame using CuPy."""
        import cupy as cp
        
        # Transfer to GPU
        gpu_frame = cp.asarray(frame, dtype=cp.uint8)
        
        # Allocate output
        output_frame = cp.zeros_like(gpu_frame)
        
        # Launch kernel
        total_pixels = frame.shape[0] * frame.shape[1]
        threads_per_block = 256
        blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block
        
        self.kernels['bgr_to_rgb'](
            (blocks_per_grid,), (threads_per_block,),
            (gpu_frame, output_frame, frame.shape[1], frame.shape[0], frame.shape[2])
        )
        
        # Synchronize and return
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(output_frame)
    
    def _process_frame_pytorch(self, frame: np.ndarray) -> np.ndarray:
        """Process frame using PyTorch."""
        import torch
        
        # Convert to tensor
        tensor = torch.from_numpy(frame).cuda().permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Apply processing (example: BGR to RGB)
        processed = tensor.flip(dims=[1])  # Flip channels
        
        # Convert back
        result = processed.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = (result * 255).astype(np.uint8)
        
        return result
    
    def _process_frame_cpu_fallback(self, frame: np.ndarray) -> np.ndarray:
        """CPU fallback for frame processing."""
        try:
            import cv2
            # Simple BGR to RGB conversion
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        except ImportError:
            # Ultimate fallback - just flip channels
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                return frame[:, :, [2, 1, 0]]
            return frame
    
    def process_frame_batch_gpu(self, frames: np.ndarray) -> np.ndarray:
        """Process multiple frames on GPU."""
        if not self.cuda_available or frames.shape[0] == 0:
            return self._process_batch_cpu_fallback(frames)
        
        with self.lock:
            try:
                start_time = time.time()
                
                with self.gpu_context():
                    if self.cupy_available and self.kernels_compiled:
                        result = self._process_batch_cupy(frames)
                    elif self.pytorch_available:
                        result = self._process_batch_pytorch(frames)
                    else:
                        result = self._process_batch_cpu_fallback(frames)
                
                processing_time = (time.time() - start_time) * 1000
                self._update_performance_metrics(processing_time)
                
                return result
                
            except Exception as e:
                print(f"GPU batch processing failed: {e}")
                return self._process_batch_cpu_fallback(frames)
    
    def _process_batch_cupy(self, frames: np.ndarray) -> np.ndarray:
        """Process batch of frames using CuPy."""
        import cupy as cp
        
        batch_size, height, width, channels = frames.shape
        
        # Transfer batch to GPU
        gpu_frames = cp.asarray(frames, dtype=cp.uint8)
        output_frames = cp.zeros_like(gpu_frames)
        
        # Process each frame
        for i in range(batch_size):
            total_pixels = height * width
            threads_per_block = 256
            blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block
            
            self.kernels['bgr_to_rgb'](
                (blocks_per_grid,), (threads_per_block,),
                (gpu_frames[i], output_frames[i], width, height, channels)
            )
        
        # Synchronize and return
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(output_frames)
    
    def _process_batch_pytorch(self, frames: np.ndarray) -> np.ndarray:
        """Process batch of frames using PyTorch."""
        import torch
        
        # Convert to tensor
        tensor = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float() / 255.0
        
        # Apply batch processing (BGR to RGB)
        processed = tensor.flip(dims=[1])
        
        # Convert back
        result = processed.permute(0, 2, 3, 1).cpu().numpy()
        result = (result * 255).astype(np.uint8)
        
        return result
    
    def _process_batch_cpu_fallback(self, frames: np.ndarray) -> np.ndarray:
        """CPU fallback for batch processing."""
        try:
            import cv2
            processed = []
            for frame in frames:
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    processed.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    processed.append(frame)
            return np.array(processed)
        except ImportError:
            # Ultimate fallback
            if len(frames.shape) == 4 and frames.shape[3] == 3:
                return frames[:, :, :, [2, 1, 0]]
            return frames
    
    def resize_batch_gpu(self, frames: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize batch of frames on GPU."""
        if not self.cuda_available:
            return self._resize_batch_cpu(frames, target_size)
        
        with self.lock:
            try:
                with self.gpu_context():
                    if self.cupy_available and self.kernels_compiled:
                        return self._resize_batch_cupy(frames, target_size)
                    elif self.pytorch_available:
                        return self._resize_batch_pytorch(frames, target_size)
                    else:
                        return self._resize_batch_cpu(frames, target_size)
            except Exception as e:
                print(f"GPU resize failed: {e}")
                return self._resize_batch_cpu(frames, target_size)
    
    def _resize_batch_cupy(self, frames: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frames using CuPy kernel."""
        import cupy as cp
        
        batch_size, in_height, in_width, channels = frames.shape
        out_width, out_height = target_size
        
        # Allocate GPU memory
        gpu_input = cp.asarray(frames, dtype=cp.uint8)
        gpu_output = cp.zeros((batch_size, out_height, out_width, channels), dtype=cp.uint8)
        
        # Launch resize kernel for each frame
        threads_per_block = (16, 16)
        blocks_per_grid = (
            (out_width + threads_per_block[0] - 1) // threads_per_block[0],
            (out_height + threads_per_block[1] - 1) // threads_per_block[1]
        )
        
        for i in range(batch_size):
            self.kernels['resize_bilinear'](
                blocks_per_grid, threads_per_block,
                (gpu_input[i], gpu_output[i], in_width, in_height, 
                 out_width, out_height, channels)
            )
        
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(gpu_output)
    
    def _resize_batch_pytorch(self, frames: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frames using PyTorch."""
        import torch
        import torch.nn.functional as F
        
        # Convert to tensor
        tensor = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float()
        
        # Resize
        resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        
        # Convert back
        result = resized.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return result
    
    def _resize_batch_cpu(self, frames: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """CPU fallback for resizing."""
        try:
            import cv2
            resized = []
            for frame in frames:
                resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                resized.append(resized_frame)
            return np.array(resized)
        except ImportError:
            # Simple nearest neighbor fallback
            batch_size, height, width, channels = frames.shape
            out_width, out_height = target_size
            
            scale_x = width / out_width
            scale_y = height / out_height
            
            resized = np.zeros((batch_size, out_height, out_width, channels), dtype=frames.dtype)
            
            for b in range(batch_size):
                for y in range(out_height):
                    for x in range(out_width):
                        src_x = min(int(x * scale_x), width - 1)
                        src_y = min(int(y * scale_y), height - 1)
                        resized[b, y, x] = frames[b, src_y, src_x]
            
            return resized
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance tracking metrics."""
        self.processing_times.append(processing_time)
        
        # Keep only recent times
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
        
        # Update GPU utilization if available
        if self.cuda_available:
            try:
                if self.pytorch_available:
                    import torch
                    memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    self.memory_usage.append(memory_used)
                    
                    if len(self.memory_usage) > 100:
                        self.memory_usage = self.memory_usage[-100:]
                        
            except Exception:
                pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        avg_memory_usage = np.mean(self.memory_usage) if self.memory_usage else 0
        
        stats = {
            'cuda_available': self.cuda_available,
            'cupy_available': self.cupy_available,
            'pytorch_available': self.pytorch_available,
            'kernels_compiled': self.kernels_compiled,
            'average_processing_time_ms': avg_processing_time,
            'average_memory_usage': avg_memory_usage,
            'memory_limit_mb': self.memory_limit / (1024**2) if self.memory_limit else None,
            'total_processed_frames': len(self.processing_times)
        }
        
        # Add GPU-specific stats if available
        if self.cuda_available:
            try:
                if self.pytorch_available:
                    import torch
                    stats.update({
                        'gpu_name': torch.cuda.get_device_name(0),
                        'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
                        'gpu_memory_allocated': torch.cuda.memory_allocated(),
                        'gpu_memory_cached': torch.cuda.memory_reserved()
                    })
            except Exception:
                pass
        
        return stats
    
    def benchmark_performance(self, num_frames: int = 100) -> Dict[str, float]:
        """Benchmark GPU performance."""
        print(f"Benchmarking GPU processor ({num_frames} frames)...")
        
        # Generate test data
        test_frames = np.random.randint(0, 255, (num_frames, 512, 512, 3), dtype=np.uint8)
        
        # Warm up
        for _ in range(5):
            self.process_frame_gpu(test_frames[0])
        
        # Clear metrics
        self.processing_times.clear()
        
        # Benchmark single frame processing
        start_time = time.time()
        for i in range(min(50, num_frames)):  # Test subset for single frames
            self.process_frame_gpu(test_frames[i])
        single_frame_time = time.time() - start_time
        
        # Benchmark batch processing
        batch_sizes = [5, 10, 20] if num_frames >= 20 else [min(5, num_frames)]
        batch_results = {}
        
        for batch_size in batch_sizes:
            if num_frames >= batch_size:
                self._cleanup_gpu_memory()  # Clean before each test
                
                start_time = time.time()
                for i in range(0, min(num_frames, 50), batch_size):
                    end_idx = min(i + batch_size, num_frames)
                    batch = test_frames[i:end_idx]
                    self.process_frame_batch_gpu(batch)
                
                batch_time = time.time() - start_time
                batch_results[f'batch_{batch_size}'] = batch_time
        
        results = {
            'single_frame_total_time': single_frame_time,
            'single_frame_fps': min(50, num_frames) / single_frame_time,
            'average_frame_time_ms': np.mean(self.processing_times) if self.processing_times else 0,
            'cuda_available': self.cuda_available,
            'cupy_available': self.cupy_available,
            'pytorch_available': self.pytorch_available,
            **batch_results
        }
        
        print(f"GPU Benchmark Results:")
        print(f"  Single frame FPS: {results['single_frame_fps']:.1f}")
        print(f"  Average time per frame: {results['average_frame_time_ms']:.2f}ms")
        print(f"  CUDA available: {self.cuda_available}")
        
        return results
    
    def cleanup(self):
        """Clean up GPU resources."""
        with self.lock:
            self._cleanup_gpu_memory()
            
            # Clear caches
            self.processing_times.clear()
            self.memory_usage.clear()
            self.gpu_utilization.clear()
            
            # Force garbage collection
            gc.collect()
            
            print("GPU processor cleaned up")
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass