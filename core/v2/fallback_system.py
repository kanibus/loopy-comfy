# -*- coding: utf-8 -*-
"""
Fallback Chain System for LoopyComfy v2.0

This module provides automatic degradation when advanced features are unavailable,
ensuring the system never crashes and always provides functionality.
"""

import time
import importlib.util
from typing import Optional, Any, List, Dict
from abc import ABC, abstractmethod


class FeatureFallbackChain:
    """Automatic degradation when features unavailable."""
    
    def __init__(self, config, resource_monitor):
        """Initialize fallback chain."""
        self.config = config
        self.monitor = resource_monitor
        self.fallback_history = []
        
        # Cache for loaded components
        self._processor_cache = {}
        self._engine_cache = {}
    
    def get_processor(self):
        """Get best available processor with fallback chain."""
        recommendations = self.monitor.get_recommended_quality()
        
        # Try GPU processor first if enabled and recommended
        if (self.config.is_feature_enabled('gpu') and 
            recommendations.get('gpu_enabled', False)):
            try:
                processor = self._load_gpu_processor()
                if processor:
                    return processor
            except Exception as e:
                self.log_fallback("GPU processor", "Parallel CPU", str(e))
        
        # Try parallel CPU processor if recommended
        if recommendations.get('parallel_workers', 1) > 1:
            try:
                processor = self._load_parallel_processor(
                    recommendations['parallel_workers']
                )
                if processor:
                    return processor
            except Exception as e:
                self.log_fallback("Parallel CPU", "Basic", str(e))
        
        # Fallback to basic processor (v1.x compatibility)
        return self._load_basic_processor()
    
    def get_ml_engine(self):
        """Get ML engine with appropriate quality."""
        if not self.config.is_feature_enabled('ml'):
            return self.get_basic_markov_engine()
        
        recommendations = self.monitor.get_recommended_quality()
        ml_quality = recommendations.get('ml_quality', 'cpu')
        
        try:
            engine = self._load_ml_engine(ml_quality)
            
            # Test performance if ML is enabled
            if engine and not self._test_ml_performance(engine, ml_quality):
                # Try downgrading quality
                if ml_quality == 'full':
                    engine = self._load_ml_engine('lite')
                    if not self._test_ml_performance(engine, 'lite'):
                        engine = self._load_ml_engine('cpu')
                elif ml_quality == 'lite':
                    engine = self._load_ml_engine('cpu')
            
            if engine:
                return engine
                
        except Exception as e:
            self.log_fallback("ML Engine", "Basic Markov", str(e))
        
        return self.get_basic_markov_engine()
    
    def get_basic_markov_engine(self):
        """Get v1.x Markov engine for compatibility."""
        if 'basic_markov' not in self._engine_cache:
            try:
                from core.markov_engine import MarkovTransitionEngine
                self._engine_cache['basic_markov'] = MarkovTransitionEngine()
            except Exception as e:
                raise RuntimeError(f"Failed to load basic Markov engine: {e}")
        
        return self._engine_cache['basic_markov']
    
    def get_websocket_server(self):
        """Get WebSocket server with fallback options."""
        if not self.config.is_feature_enabled('real_time'):
            return None
        
        try:
            # Try enhanced WebSocket server
            server = self._load_websocket_server()
            if server:
                return server
        except Exception as e:
            self.log_fallback("Enhanced WebSocket", "Basic HTTP", str(e))
        
        # Fallback to basic HTTP streaming
        try:
            return self._load_http_streaming_server()
        except Exception as e:
            self.log_fallback("HTTP Streaming", "Disabled", str(e))
        
        return None
    
    def _load_gpu_processor(self):
        """Load GPU accelerated processor."""
        if 'gpu_processor' not in self._processor_cache:
            try:
                spec = importlib.util.spec_from_file_location(
                    "gpu_processor", 
                    "advanced/gpu_processor.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    self._processor_cache['gpu_processor'] = module.GPUAcceleratedProcessor(
                        self.config, self.monitor
                    )
                else:
                    return None
            except Exception:
                return None
        
        return self._processor_cache.get('gpu_processor')
    
    def _load_parallel_processor(self, workers: int):
        """Load parallel CPU processor."""
        cache_key = f'parallel_processor_{workers}'
        
        if cache_key not in self._processor_cache:
            try:
                spec = importlib.util.spec_from_file_location(
                    "parallel_processor", 
                    "advanced/parallel_processor.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    self._processor_cache[cache_key] = module.ParallelCPUProcessor(workers)
                else:
                    return None
            except Exception:
                return None
        
        return self._processor_cache.get(cache_key)
    
    def _load_basic_processor(self):
        """Load basic processor (v1.x compatibility)."""
        if 'basic_processor' not in self._processor_cache:
            # Use existing v1.x implementation
            class BasicProcessor:
                """Basic processor using v1.x methods."""
                
                def process_frame_batch(self, frames):
                    """Process frames using basic methods."""
                    import cv2
                    import numpy as np
                    
                    processed = []
                    for frame in frames:
                        # Basic processing - just ensure RGB format
                        if len(frame.shape) == 3:
                            if frame.shape[2] == 3:
                                # Assume BGR, convert to RGB
                                processed.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            else:
                                processed.append(frame)
                        else:
                            processed.append(frame)
                    
                    return np.array(processed)
                
                def cleanup(self):
                    """No cleanup needed for basic processor."""
                    pass
            
            self._processor_cache['basic_processor'] = BasicProcessor()
        
        return self._processor_cache['basic_processor']
    
    def _load_ml_engine(self, quality: str):
        """Load ML engine with specified quality."""
        cache_key = f'ml_engine_{quality}'
        
        if cache_key not in self._engine_cache:
            try:
                spec = importlib.util.spec_from_file_location(
                    "ml_engine", 
                    "advanced/ml_transition.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    self._engine_cache[cache_key] = module.MLEnhancedMarkovEngine(
                        quality_mode=quality
                    )
                else:
                    return None
            except Exception:
                return None
        
        return self._engine_cache.get(cache_key)
    
    def _load_websocket_server(self):
        """Load WebSocket server."""
        try:
            spec = importlib.util.spec_from_file_location(
                "websocket_server", 
                "advanced/websocket_server.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                return module.WebSocketPreviewServer(self.config)
            else:
                return None
        except Exception:
            return None
    
    def _load_http_streaming_server(self):
        """Load HTTP streaming fallback."""
        try:
            spec = importlib.util.spec_from_file_location(
                "http_streaming", 
                "advanced/http_streaming.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                return module.HTTPStreamingServer(self.config)
            else:
                return None
        except Exception:
            return None
    
    def _test_ml_performance(self, engine, quality: str) -> bool:
        """Test if ML inference meets performance targets."""
        try:
            import numpy as np
            
            # Create test frames
            test_frame = np.random.rand(224, 224, 3).astype(np.float32)
            
            # Measure inference time
            start = time.time()
            engine.predict_transition_quality(test_frame, test_frame)
            inference_time = (time.time() - start) * 1000  # ms
            
            target = self.config.get(f'ml.max_inference_ms.{quality}', 100)
            performance_ok = inference_time < target
            
            if not performance_ok:
                print(f"ML inference too slow: {inference_time:.1f}ms (target: {target}ms)")
            
            return performance_ok
            
        except Exception as e:
            print(f"ML performance test failed: {e}")
            return False
    
    def log_fallback(self, from_feature: str, to_feature: str, reason: str):
        """Log fallback for debugging and user awareness."""
        self.fallback_history.append({
            'from': from_feature,
            'to': to_feature,
            'reason': reason,
            'timestamp': time.time()
        })
        
        print(f"[FALLBACK] {from_feature} → {to_feature} ({reason})")
        
        # Notify user about fallback
        if self.config.get('monitoring.enabled', True):
            self._notify_user_fallback(from_feature, to_feature, reason)
    
    def _notify_user_fallback(self, from_feature: str, to_feature: str, reason: str):
        """Notify user about fallback in UI."""
        # This would integrate with the UI notification system
        notification = {
            'type': 'fallback',
            'message': f'Switched from {from_feature} to {to_feature}',
            'reason': reason,
            'severity': 'warning' if to_feature != 'Disabled' else 'error',
            'timestamp': time.time()
        }
        
        # Store for UI to pick up
        if not hasattr(self, '_notifications'):
            self._notifications = []
        
        self._notifications.append(notification)
        
        # Keep only last 10 notifications
        self._notifications = self._notifications[-10:]
    
    def get_fallback_history(self) -> List[Dict]:
        """Get history of fallbacks for debugging."""
        return self.fallback_history.copy()
    
    def get_recent_notifications(self) -> List[Dict]:
        """Get recent notifications for UI."""
        return getattr(self, '_notifications', [])
    
    def clear_notifications(self):
        """Clear notifications after UI has read them."""
        self._notifications = []
    
    def get_current_capabilities(self) -> Dict[str, Any]:
        """Get current system capabilities after fallbacks."""
        capabilities = {
            'processor_type': 'basic',
            'ml_enabled': False,
            'gpu_enabled': False,
            'real_time_enabled': False,
            'parallel_workers': 1,
            'ml_quality': None
        }
        
        # Determine processor type
        if 'gpu_processor' in self._processor_cache:
            capabilities['processor_type'] = 'gpu'
            capabilities['gpu_enabled'] = True
        elif any('parallel_processor' in key for key in self._processor_cache.keys()):
            capabilities['processor_type'] = 'parallel_cpu'
            capabilities['parallel_workers'] = max(2, len([k for k in self._processor_cache.keys() if 'parallel_processor' in k]))
        
        # Check ML capabilities
        ml_engines = [k for k in self._engine_cache.keys() if k.startswith('ml_engine')]
        if ml_engines:
            capabilities['ml_enabled'] = True
            # Get highest quality available
            if 'ml_engine_full' in self._engine_cache:
                capabilities['ml_quality'] = 'full'
            elif 'ml_engine_lite' in self._engine_cache:
                capabilities['ml_quality'] = 'lite'
            elif 'ml_engine_cpu' in self._engine_cache:
                capabilities['ml_quality'] = 'cpu'
        
        # Check real-time capabilities
        capabilities['real_time_enabled'] = self.get_websocket_server() is not None
        
        return capabilities
    
    def cleanup(self):
        """Clean up all cached components."""
        for processor in self._processor_cache.values():
            if hasattr(processor, 'cleanup'):
                try:
                    processor.cleanup()
                except Exception as e:
                    print(f"Cleanup error: {e}")
        
        self._processor_cache.clear()
        self._engine_cache.clear()
        
        print("Fallback system cleaned up")