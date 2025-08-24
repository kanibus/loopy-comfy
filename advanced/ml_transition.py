# -*- coding: utf-8 -*-
"""
ML-Enhanced Transition Detection for LoopyComfy v2.0

This module provides adaptive quality ML models for improved transition
detection with multiple performance tiers and hardware optimization.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import time
import warnings
from typing import Optional, Tuple, Dict, Any, List
from contextlib import contextmanager
import threading
import queue


class AdaptiveTransitionModel:
    """ML model with multiple quality modes for different hardware."""
    
    def __init__(self, quality_mode: str = 'auto', device: str = 'auto', cache_size: int = 1000):
        """Initialize adaptive ML model."""
        self.quality_mode = quality_mode
        self.device = self._detect_device(device)
        self.cache_size = cache_size
        
        # Model components
        self.model = None
        self.transform = None
        self.feature_cache = {}
        
        # Performance tracking
        self.inference_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Load model
        self._load_model()
        
        print(f"ML Transition Model initialized: {quality_mode} on {self.device}")
    
    def _detect_device(self, device_preference: str) -> torch.device:
        """Detect best available device."""
        if device_preference == 'auto':
            if torch.cuda.is_available():
                # Check if we have enough GPU memory
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    if gpu_memory > 2 * (1024**3):  # 2GB minimum
                        return torch.device('cuda')
                except Exception:
                    pass
            return torch.device('cpu')
        return torch.device(device_preference)
    
    def _load_model(self):
        """Load appropriate model based on quality mode."""
        if self.quality_mode == 'auto':
            self.quality_mode = self._auto_select_quality()
        
        if self.quality_mode == 'full':
            self.model = self._create_full_model()
        elif self.quality_mode == 'lite':
            self.model = self._create_lite_model()
        else:  # cpu
            self.model = self._create_cpu_model()
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup transform
        self.transform = self._create_transform()
        
        # Warm up model
        self._warmup_model()
    
    def _auto_select_quality(self) -> str:
        """Automatically select quality based on hardware."""
        if self.device.type == 'cuda':
            try:
                # Check GPU memory
                gpu_props = torch.cuda.get_device_properties(0)
                mem_gb = gpu_props.total_memory / (1024**3)
                
                if mem_gb >= 8:
                    return 'full'
                elif mem_gb >= 4:
                    return 'lite'
            except Exception:
                pass
        return 'cpu'
    
    def _create_full_model(self) -> nn.Module:
        """Create full-quality ResNet50 model."""
        class FullTransitionModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Use pretrained ResNet50 backbone
                resnet = models.resnet50(pretrained=True)
                self.backbone = nn.Sequential(*list(resnet.children())[:-1])
                
                # Freeze backbone
                for param in self.backbone.parameters():
                    param.requires_grad = False
                
                # Transition quality head
                self.quality_head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(2048 * 2, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 3),  # [poor, good, excellent]
                    nn.Softmax(dim=1)
                )
            
            def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
                # Extract features
                with torch.no_grad():
                    feat1 = self.backbone(frame1)
                    feat2 = self.backbone(frame2)
                
                # Combine features
                combined = torch.cat([feat1, feat2], dim=1)
                
                # Predict quality
                return self.quality_head(combined)
        
        return FullTransitionModel()
    
    def _create_lite_model(self) -> nn.Module:
        """Create lightweight MobileNetV3 model."""
        class LiteTransitionModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Use MobileNetV3 small
                mobilenet = models.mobilenet_v3_small(pretrained=True)
                self.features = mobilenet.features
                
                # Freeze features
                for param in self.features.parameters():
                    param.requires_grad = False
                
                # Adaptive pooling
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                
                # Quality head
                self.quality_head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(576 * 2, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 3),
                    nn.Softmax(dim=1)
                )
            
            def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
                # Extract features
                with torch.no_grad():
                    feat1 = self.avgpool(self.features(frame1)).flatten(1)
                    feat2 = self.avgpool(self.features(frame2)).flatten(1)
                
                # Combine and predict
                combined = torch.cat([feat1, feat2], dim=1)
                return self.quality_head(combined)
        
        return LiteTransitionModel()
    
    def _create_cpu_model(self) -> nn.Module:
        """Create minimal model for CPU inference."""
        class CPUTransitionModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Simple CNN for CPU
                self.features = nn.Sequential(
                    # First block
                    nn.Conv2d(3, 32, 7, stride=2, padding=3),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    
                    # Second block
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    
                    # Third block
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    # Global pooling
                    nn.AdaptiveAvgPool2d(1)
                )
                
                # Quality head
                self.quality_head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 2, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(128, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 3),
                    nn.Softmax(dim=1)
                )
            
            def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
                feat1 = self.features(frame1).flatten(1)
                feat2 = self.features(frame2).flatten(1)
                combined = torch.cat([feat1, feat2], dim=1)
                return self.quality_head(combined)
        
        return CPUTransitionModel()
    
    def _create_transform(self) -> transforms.Compose:
        """Create appropriate transform for the model."""
        if self.quality_mode == 'cpu':
            # Smaller input size for CPU
            size = 128
        else:
            # Standard ImageNet size
            size = 224
        
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _warmup_model(self):
        """Warm up model with dummy inference."""
        try:
            dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Run a few warmup predictions
            for _ in range(3):
                _ = self.predict(dummy_frame, dummy_frame)
            
            print(f"Model warmed up successfully ({self.quality_mode})")
            
        except Exception as e:
            print(f"Model warmup failed: {e}")
    
    def _create_cache_key(self, frame1: np.ndarray, frame2: np.ndarray) -> str:
        """Create cache key for frame pair."""
        # Use hash of frame data for caching
        hash1 = hash(frame1.tobytes())
        hash2 = hash(frame2.tobytes())
        return f"{hash1}_{hash2}"
    
    @torch.no_grad()
    def predict(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Predict transition quality with caching."""
        with self.lock:
            # Check cache first
            cache_key = self._create_cache_key(frame1, frame2)
            
            if cache_key in self.feature_cache:
                self.cache_hits += 1
                return self.feature_cache[cache_key]
            
            self.cache_misses += 1
            
            try:
                start_time = time.time()
                
                # Prepare inputs
                tensor1 = self._prepare_input(frame1)
                tensor2 = self._prepare_input(frame2)
                
                # Forward pass
                scores = self.model(tensor1, tensor2)
                
                # Convert to numpy
                result = scores.cpu().numpy()[0]
                
                # Track inference time
                inference_time = (time.time() - start_time) * 1000  # ms
                self.inference_times.append(inference_time)
                
                # Keep only recent times
                if len(self.inference_times) > 100:
                    self.inference_times = self.inference_times[-100:]
                
                # Cache result if space available
                if len(self.feature_cache) < self.cache_size:
                    self.feature_cache[cache_key] = result.copy()
                elif len(self.feature_cache) >= self.cache_size:
                    # Remove oldest entry (simple LRU approximation)
                    oldest_key = next(iter(self.feature_cache))
                    del self.feature_cache[oldest_key]
                    self.feature_cache[cache_key] = result.copy()
                
                return result
                
            except Exception as e:
                print(f"ML prediction failed: {e}")
                # Return neutral scores as fallback
                return np.array([0.33, 0.34, 0.33], dtype=np.float32)
    
    def _prepare_input(self, frame: np.ndarray) -> torch.Tensor:
        """Prepare input frame for model."""
        try:
            # Ensure correct format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Apply transform
            tensor = self.transform(frame)
            
            # Add batch dimension and move to device
            tensor = tensor.unsqueeze(0).to(self.device, non_blocking=True)
            
            return tensor
            
        except Exception as e:
            print(f"Input preparation failed: {e}")
            # Return dummy tensor
            dummy_size = 128 if self.quality_mode == 'cpu' else 224
            return torch.zeros(1, 3, dummy_size, dummy_size, device=self.device)
    
    def predict_batch(self, frame_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """Predict transition quality for multiple frame pairs."""
        results = []
        
        for frame1, frame2 in frame_pairs:
            result = self.predict(frame1, frame2)
            results.append(result)
        
        return results
    
    def get_average_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self.feature_cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def clear_cache(self):
        """Clear prediction cache."""
        with self.lock:
            self.feature_cache.clear()
            print("ML prediction cache cleared")
    
    def optimize_model(self):
        """Optimize model for inference if possible."""
        try:
            if self.device.type == 'cuda':
                # Try TensorRT optimization if available
                try:
                    import torch_tensorrt
                    self.model = torch_tensorrt.compile(
                        self.model,
                        inputs=[
                            torch_tensorrt.Input((1, 3, 224, 224)),
                            torch_tensorrt.Input((1, 3, 224, 224))
                        ],
                        enabled_precisions={torch.float16}
                    )
                    print("Model optimized with TensorRT")
                except ImportError:
                    pass
            
            # Try TorchScript optimization
            try:
                dummy_input1 = torch.randn(1, 3, 224, 224, device=self.device)
                dummy_input2 = torch.randn(1, 3, 224, 224, device=self.device)
                
                traced_model = torch.jit.trace(self.model, (dummy_input1, dummy_input2))
                traced_model = torch.jit.optimize_for_inference(traced_model)
                
                self.model = traced_model
                print("Model optimized with TorchScript")
                
            except Exception as e:
                print(f"TorchScript optimization failed: {e}")
        
        except Exception as e:
            print(f"Model optimization failed: {e}")
    
    def benchmark_performance(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance."""
        print(f"Benchmarking ML model performance ({num_iterations} iterations)...")
        
        # Generate test data
        dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Warm up
        for _ in range(10):
            self.predict(dummy_frame, dummy_frame)
        
        # Clear timing history
        self.inference_times.clear()
        
        # Benchmark
        start_time = time.time()
        
        for i in range(num_iterations):
            # Vary input slightly to avoid cache hits
            frame1 = dummy_frame.copy()
            frame2 = dummy_frame.copy()
            frame1[0, 0, 0] = i % 255
            frame2[0, 0, 1] = (i + 1) % 255
            
            self.predict(frame1, frame2)
        
        total_time = time.time() - start_time
        
        results = {
            'total_time_seconds': total_time,
            'average_time_ms': self.get_average_inference_time(),
            'throughput_fps': num_iterations / total_time,
            'quality_mode': self.quality_mode,
            'device': str(self.device),
            'iterations': num_iterations
        }
        
        print(f"Benchmark results:")
        print(f"  Average inference time: {results['average_time_ms']:.2f}ms")
        print(f"  Throughput: {results['throughput_fps']:.1f} FPS")
        print(f"  Quality mode: {results['quality_mode']}")
        
        return results


class MLEnhancedMarkovEngine:
    """Markov engine enhanced with ML transition quality scoring."""
    
    def __init__(self, quality_mode: str = 'auto', config: Optional[Dict] = None):
        """Initialize ML-enhanced Markov engine."""
        self.config = config or {}
        
        # Initialize ML model
        try:
            self.ml_model = AdaptiveTransitionModel(
                quality_mode=quality_mode,
                cache_size=self.config.get('cache_size', 1000)
            )
            self.ml_enabled = True
        except Exception as e:
            print(f"Failed to initialize ML model: {e}")
            self.ml_model = None
            self.ml_enabled = False
        
        # Fallback to basic Markov engine
        from core.markov_engine import MarkovTransitionEngine
        self.basic_engine = MarkovTransitionEngine()
        
        # Performance tracking
        self.ml_predictions = 0
        self.fallback_predictions = 0
        
        print(f"ML-Enhanced Markov Engine initialized (ML: {'enabled' if self.ml_enabled else 'disabled'})")
    
    def predict_transition_quality(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Predict transition quality between frames."""
        try:
            if self.ml_enabled and self.ml_model:
                self.ml_predictions += 1
                return self.ml_model.predict(frame1, frame2)
            else:
                # Fallback to heuristic scoring
                return self._heuristic_transition_quality(frame1, frame2)
                
        except Exception as e:
            print(f"ML prediction failed: {e}")
            return self._heuristic_transition_quality(frame1, frame2)
    
    def _heuristic_transition_quality(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Heuristic-based transition quality as ML fallback."""
        try:
            self.fallback_predictions += 1
            
            # Simple heuristics based on frame similarity
            import cv2
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            
            # Calculate structural similarity
            from skimage.metrics import structural_similarity as ssim
            similarity = ssim(gray1, gray2)
            
            # Convert similarity to quality scores
            if similarity > 0.8:
                # High similarity - poor transition
                return np.array([0.7, 0.2, 0.1])  # [poor, good, excellent]
            elif similarity > 0.6:
                # Medium similarity - good transition
                return np.array([0.2, 0.7, 0.1])
            else:
                # Low similarity - excellent transition
                return np.array([0.1, 0.2, 0.7])
                
        except Exception as e:
            print(f"Heuristic scoring failed: {e}")
            # Return neutral scores
            return np.array([0.33, 0.34, 0.33])
    
    def get_next_state(self, current_state: int, video_metadata: List[Dict]) -> int:
        """Get next state with ML-enhanced selection."""
        try:
            # Use basic Markov logic first
            candidates = self.basic_engine.get_candidate_states(current_state, video_metadata)
            
            if not candidates or not self.ml_enabled:
                return self.basic_engine.get_next_state(current_state, video_metadata)
            
            # Score transitions with ML if we have frame data
            best_candidate = current_state
            best_score = 0.0
            
            current_frame = self._get_last_frame(current_state, video_metadata)
            
            for candidate in candidates:
                candidate_frame = self._get_first_frame(candidate, video_metadata)
                
                if current_frame is not None and candidate_frame is not None:
                    quality_scores = self.predict_transition_quality(current_frame, candidate_frame)
                    # Use 'excellent' score (index 2) as selection criterion
                    transition_score = quality_scores[2]
                    
                    if transition_score > best_score:
                        best_score = transition_score
                        best_candidate = candidate
            
            return best_candidate
            
        except Exception as e:
            print(f"ML-enhanced state selection failed: {e}")
            return self.basic_engine.get_next_state(current_state, video_metadata)
    
    def _get_last_frame(self, video_idx: int, video_metadata: List[Dict]) -> Optional[np.ndarray]:
        """Get last frame of video (placeholder - would load actual frame)."""
        # This would integrate with video loading system
        # For now, return None to fall back to basic Markov
        return None
    
    def _get_first_frame(self, video_idx: int, video_metadata: List[Dict]) -> Optional[np.ndarray]:
        """Get first frame of video (placeholder - would load actual frame)."""
        # This would integrate with video loading system
        # For now, return None to fall back to basic Markov
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for ML engine."""
        stats = {
            'ml_enabled': self.ml_enabled,
            'ml_predictions': self.ml_predictions,
            'fallback_predictions': self.fallback_predictions,
            'total_predictions': self.ml_predictions + self.fallback_predictions
        }
        
        if self.ml_enabled and self.ml_model:
            stats.update({
                'average_inference_time_ms': self.ml_model.get_average_inference_time(),
                'cache_stats': self.ml_model.get_cache_stats(),
                'quality_mode': self.ml_model.quality_mode,
                'device': str(self.ml_model.device)
            })
        
        return stats
    
    def optimize_for_performance(self):
        """Optimize ML model for better performance."""
        if self.ml_enabled and self.ml_model:
            self.ml_model.optimize_model()
    
    def benchmark(self, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark ML performance."""
        if self.ml_enabled and self.ml_model:
            return self.ml_model.benchmark_performance(iterations)
        else:
            return {'error': 'ML model not available for benchmarking'}
    
    def cleanup(self):
        """Clean up resources."""
        if self.ml_model:
            self.ml_model.clear_cache()
        
        print("ML-Enhanced Markov Engine cleaned up")