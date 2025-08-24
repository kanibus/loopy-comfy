# -*- coding: utf-8 -*-
"""
Resource Monitoring and Adaptive System for LoopyComfy v2.0

This module provides system resource monitoring with adaptive quality settings
and emergency cleanup capabilities.
"""

import psutil
import warnings
import time
import gc
from contextlib import contextmanager
from typing import Dict, Any, Optional


class ResourceMonitor:
    """Monitor system resources and adapt quality settings."""
    
    def __init__(self, config):
        """Initialize resource monitor."""
        self.config = config
        self.memory_limit = config.get('performance.memory_limit_mb')
        self.warning_threshold = config.get('performance.memory_warning_threshold', 0.9)
        
        # Initialize GPU monitoring if available
        self.gpu_available = False
        self.gpu_handle = None
        
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_available = True
            self.pynvml = pynvml
            print("GPU monitoring initialized")
        except Exception as e:
            print(f"GPU monitoring not available: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system resource status."""
        # CPU and Memory info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        status = {
            'cpu_percent': cpu_percent,
            'memory': {
                'total': memory_info.total,
                'available': memory_info.available,
                'percent': memory_info.percent,
                'used': memory_info.used
            },
            'process': {
                'memory_rss': psutil.Process().memory_info().rss,
                'memory_percent': psutil.Process().memory_percent(),
                'cpu_percent': psutil.Process().cpu_percent()
            },
            'gpu': None,
            'timestamp': time.time()
        }
        
        # GPU info if available
        if self.gpu_available and self.gpu_handle:
            try:
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                util_info = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                temp_info = self.pynvml.nvmlDeviceGetTemperature(self.gpu_handle, self.pynvml.NVML_TEMPERATURE_GPU)
                
                status['gpu'] = {
                    'memory_total': mem_info.total,
                    'memory_used': mem_info.used,
                    'memory_free': mem_info.free,
                    'memory_percent': (mem_info.used / mem_info.total) * 100,
                    'utilization': util_info.gpu,
                    'memory_utilization': util_info.memory,
                    'temperature': temp_info
                }
            except Exception as e:
                print(f"GPU monitoring error: {e}")
                status['gpu'] = {'error': str(e)}
        
        return status
    
    def get_recommended_quality(self) -> Dict[str, Any]:
        """Determine recommended quality settings based on resources."""
        status = self.get_system_status()
        
        recommendations = {
            'ml_quality': 'cpu',
            'gpu_enabled': False,
            'parallel_workers': 2,
            'cache_level': 'basic',
            'preview_quality': 'low',
            'batch_size': 5,
            'real_time_capable': False
        }
        
        # Memory-based recommendations
        available_gb = status['memory']['available'] / (1024**3)
        used_percent = status['memory']['percent']
        
        if available_gb > 16 and used_percent < 60:
            recommendations.update({
                'cache_level': 'intelligent',
                'preview_quality': 'high',
                'parallel_workers': min(psutil.cpu_count(), 8),
                'batch_size': 20,
                'real_time_capable': True
            })
        elif available_gb > 8 and used_percent < 70:
            recommendations.update({
                'cache_level': 'intelligent',
                'preview_quality': 'medium',
                'parallel_workers': max(2, psutil.cpu_count() // 2),
                'batch_size': 10,
                'real_time_capable': True
            })
        elif available_gb > 4 and used_percent < 80:
            recommendations.update({
                'cache_level': 'basic',
                'preview_quality': 'low',
                'parallel_workers': 2,
                'batch_size': 5,
                'real_time_capable': False
            })
        else:
            # Low memory situation
            recommendations.update({
                'cache_level': 'off',
                'preview_quality': 'minimal',
                'parallel_workers': 1,
                'batch_size': 2,
                'real_time_capable': False
            })
        
        # GPU-based recommendations
        if status['gpu'] and not status['gpu'].get('error'):
            gpu_mem_gb = status['gpu']['memory_free'] / (1024**3)
            gpu_util = status['gpu']['utilization']
            
            if gpu_mem_gb > 4 and gpu_util < 50:
                recommendations.update({
                    'gpu_enabled': True,
                    'ml_quality': 'full',
                    'real_time_capable': True
                })
            elif gpu_mem_gb > 2 and gpu_util < 70:
                recommendations.update({
                    'gpu_enabled': True,
                    'ml_quality': 'lite'
                })
        
        # CPU-based adjustments
        cpu_percent = status['cpu_percent']
        if cpu_percent > 80:
            recommendations['parallel_workers'] = max(1, recommendations['parallel_workers'] // 2)
            recommendations['real_time_capable'] = False
        
        return recommendations
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable range."""
        if not self.memory_limit:
            return True  # No limit set
        
        current_mb = psutil.Process().memory_info().rss / (1024**2)
        
        if current_mb > self.memory_limit * self.warning_threshold:
            warnings.warn(
                f"Memory usage ({current_mb:.0f}MB) approaching limit "
                f"({self.memory_limit}MB)", 
                ResourceWarning
            )
            
            if current_mb > self.memory_limit:
                # Trigger cleanup but don't crash
                self.emergency_cleanup()
                return False
        
        return True
    
    def emergency_cleanup(self):
        """Emergency cleanup when resources are critical."""
        print("Emergency cleanup triggered - releasing resources")
        
        # Clear Python garbage
        collected = gc.collect()
        print(f"Garbage collected: {collected} objects")
        
        # Clear PyTorch cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("PyTorch CUDA cache cleared")
        except ImportError:
            pass
        
        # Clear CuPy cache if available
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            print("CuPy memory pools cleared")
        except ImportError:
            pass
        
        # Clear OpenCV cache if available
        try:
            import cv2
            # OpenCV doesn't have explicit cache clearing
            pass
        except ImportError:
            pass
    
    @contextmanager
    def resource_guard(self, operation_name: str = "operation"):
        """Context manager for resource-intensive operations."""
        initial_memory = psutil.Process().memory_info().rss
        start_time = time.time()
        
        try:
            yield
        except Exception as e:
            print(f"Operation '{operation_name}' failed: {e}")
            raise
        finally:
            end_time = time.time()
            final_memory = psutil.Process().memory_info().rss
            
            memory_increase = (final_memory - initial_memory) / (1024**2)
            duration = end_time - start_time
            
            print(f"Operation '{operation_name}' completed:")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Memory delta: {memory_increase:+.1f}MB")
            
            if memory_increase > 1000:  # More than 1GB increase
                print(f"Warning: Large memory increase detected")
                if self.config.get('advanced_features.auto_fallback', True):
                    self.emergency_cleanup()
    
    def get_performance_score(self) -> float:
        """Calculate overall performance score (0-1)."""
        status = self.get_system_status()
        
        # CPU score (lower usage = better)
        cpu_score = max(0, (100 - status['cpu_percent']) / 100)
        
        # Memory score (more available = better)
        memory_score = status['memory']['available'] / status['memory']['total']
        
        # GPU score (if available)
        gpu_score = 1.0  # Default if no GPU
        if status['gpu'] and not status['gpu'].get('error'):
            gpu_usage = status['gpu']['utilization']
            gpu_memory_usage = status['gpu']['memory_percent']
            gpu_score = max(0, (200 - gpu_usage - gpu_memory_usage) / 200)
        
        # Weighted average
        total_score = (cpu_score * 0.3 + memory_score * 0.5 + gpu_score * 0.2)
        
        return max(0, min(1, total_score))
    
    def should_reduce_quality(self) -> bool:
        """Determine if quality should be reduced based on resources."""
        score = self.get_performance_score()
        return score < 0.3  # Reduce quality if performance is poor
    
    def get_optimization_suggestions(self) -> list:
        """Get optimization suggestions based on current state."""
        status = self.get_system_status()
        suggestions = []
        
        # Memory suggestions
        if status['memory']['percent'] > 85:
            suggestions.append("High memory usage detected - consider reducing batch size")
        
        # CPU suggestions
        if status['cpu_percent'] > 90:
            suggestions.append("High CPU usage - consider reducing parallel workers")
        
        # GPU suggestions
        if status['gpu'] and not status['gpu'].get('error'):
            if status['gpu']['memory_percent'] > 90:
                suggestions.append("GPU memory full - consider using CPU processing")
            elif status['gpu']['temperature'] > 85:
                suggestions.append("GPU running hot - consider reducing processing load")
        
        # Process-specific suggestions
        if status['process']['memory_percent'] > 50:
            suggestions.append("Process using high memory - consider enabling emergency cleanup")
        
        return suggestions