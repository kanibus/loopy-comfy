# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for LoopyComfy v2.0 Advanced Features

This module provides testing for all v2.0 components with fallback validation,
performance benchmarking, and integration testing.
"""

import pytest
import asyncio
import numpy as np
import time
import threading
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List


class TestAdvancedFeaturesConfig:
    """Test configuration management system."""
    
    def test_config_loading_with_defaults(self, temp_config_dir):
        """Test configuration loads with proper defaults."""
        from core.v2.config_manager import AdvancedFeaturesConfig
        
        config_path = os.path.join(temp_config_dir, "test_config.yaml")
        config = AdvancedFeaturesConfig(config_path)
        
        # Check defaults
        assert config.config['compatibility_mode'] == 'v1'
        assert config.config['advanced_features']['enabled'] is False
        assert config.config['advanced_features']['auto_fallback'] is True
        assert config.config['real_time']['enabled'] is False
        assert config.config['ml']['enabled'] is False
        assert config.config['gpu']['enabled'] is False
    
    def test_config_validation(self, temp_config_dir):
        """Test configuration validation and auto-correction."""
        from core.v2.config_manager import AdvancedFeaturesConfig
        
        config_path = os.path.join(temp_config_dir, "test_config.yaml")
        config = AdvancedFeaturesConfig(config_path)
        
        # Test port availability check
        original_port = config.config['real_time']['websocket_port']
        assert isinstance(original_port, int)
        assert 1000 <= original_port <= 65535
        
        # Test auth token generation
        token = config.config['real_time']['auth_token']
        assert len(token) >= 32  # Should be a proper token, not placeholder
    
    def test_config_deep_merge(self, temp_config_dir):
        """Test deep merging of configuration files."""
        from core.v2.config_manager import AdvancedFeaturesConfig
        import yaml
        
        # Create custom config
        config_path = os.path.join(temp_config_dir, "custom_config.yaml")
        custom_config = {
            'ml': {
                'enabled': True,
                'quality_mode': 'lite'
            },
            'performance': {
                'parallel_workers': 4
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(custom_config, f)
        
        config = AdvancedFeaturesConfig(config_path)
        
        # Check merging worked
        assert config.config['ml']['enabled'] is True
        assert config.config['ml']['quality_mode'] == 'lite'
        assert config.config['performance']['parallel_workers'] == 4
        
        # Check defaults are preserved
        assert config.config['compatibility_mode'] == 'v1'
        assert config.config['advanced_features']['auto_fallback'] is True
    
    def test_feature_enablement_checks(self, temp_config_dir):
        """Test feature enablement validation."""
        from core.v2.config_manager import AdvancedFeaturesConfig
        
        config_path = os.path.join(temp_config_dir, "test_config.yaml")
        config = AdvancedFeaturesConfig(config_path)
        
        # Test disabled by default
        assert config.is_feature_enabled('real_time') is False
        assert config.is_feature_enabled('ml') is False
        assert config.is_feature_enabled('gpu') is False
        
        # Test enabling v2 features
        config.enable_v2_features()
        assert config.get_compatibility_mode() == 'v2'
        assert config.is_feature_enabled('advanced_features') is True


class TestResourceMonitor:
    """Test resource monitoring and adaptive systems."""
    
    def test_system_status_collection(self, mock_config):
        """Test system status data collection."""
        from core.v2.resource_monitor import ResourceMonitor
        
        monitor = ResourceMonitor(mock_config)
        status = monitor.get_system_status()
        
        # Check required fields
        assert 'cpu_percent' in status
        assert 'memory' in status
        assert 'process' in status
        assert 'timestamp' in status
        
        # Check memory info structure
        memory = status['memory']
        assert 'total' in memory
        assert 'available' in memory
        assert 'percent' in memory
        assert 'used' in memory
        
        # Check reasonable values
        assert 0 <= status['cpu_percent'] <= 100
        assert 0 <= memory['percent'] <= 100
    
    def test_quality_recommendations(self, mock_config):
        """Test adaptive quality recommendations."""
        from core.v2.resource_monitor import ResourceMonitor
        
        monitor = ResourceMonitor(mock_config)
        recommendations = monitor.get_recommended_quality()
        
        # Check required recommendation fields
        required_fields = [
            'ml_quality', 'gpu_enabled', 'parallel_workers',
            'cache_level', 'preview_quality', 'batch_size'
        ]
        
        for field in required_fields:
            assert field in recommendations
        
        # Check reasonable values
        assert recommendations['ml_quality'] in ['cpu', 'lite', 'full']
        assert isinstance(recommendations['gpu_enabled'], bool)
        assert 1 <= recommendations['parallel_workers'] <= 16
        assert recommendations['cache_level'] in ['off', 'basic', 'intelligent']
        assert recommendations['preview_quality'] in ['minimal', 'low', 'medium', 'high']
        assert 1 <= recommendations['batch_size'] <= 50
    
    def test_memory_usage_checking(self, mock_config):
        """Test memory usage monitoring."""
        from core.v2.resource_monitor import ResourceMonitor
        
        # Set a memory limit
        mock_config.get.return_value = 1000  # 1000 MB limit
        mock_config.get.side_effect = lambda key, default=None: {
            'performance.memory_limit_mb': 1000,
            'performance.memory_warning_threshold': 0.9
        }.get(key, default)
        
        monitor = ResourceMonitor(mock_config)
        
        # Should return True for normal usage
        result = monitor.check_memory_usage()
        assert isinstance(result, bool)
    
    def test_emergency_cleanup(self, mock_config):
        """Test emergency cleanup functionality."""
        from core.v2.resource_monitor import ResourceMonitor
        
        monitor = ResourceMonitor(mock_config)
        
        # Should not raise exceptions
        monitor.emergency_cleanup()
    
    def test_performance_score_calculation(self, mock_config):
        """Test performance score calculation."""
        from core.v2.resource_monitor import ResourceMonitor
        
        monitor = ResourceMonitor(mock_config)
        score = monitor.get_performance_score()
        
        # Should return a score between 0 and 1
        assert 0.0 <= score <= 1.0
    
    def test_optimization_suggestions(self, mock_config):
        """Test optimization suggestions generation."""
        from core.v2.resource_monitor import ResourceMonitor
        
        monitor = ResourceMonitor(mock_config)
        suggestions = monitor.get_optimization_suggestions()
        
        assert isinstance(suggestions, list)
        # Suggestions might be empty if system is running well
        for suggestion in suggestions:
            assert isinstance(suggestion, str)
            assert len(suggestion) > 0


class TestFallbackSystem:
    """Test intelligent fallback chain system."""
    
    def test_fallback_chain_initialization(self, mock_config, mock_monitor):
        """Test fallback chain initializes correctly."""
        from core.v2.fallback_system import FeatureFallbackChain
        
        chain = FeatureFallbackChain(mock_config, mock_monitor)
        
        assert chain.config == mock_config
        assert chain.monitor == mock_monitor
        assert isinstance(chain.fallback_history, list)
    
    def test_processor_fallback_chain(self, mock_config, mock_monitor):
        """Test processor fallback logic."""
        from core.v2.fallback_system import FeatureFallbackChain
        
        # Mock recommendations
        mock_monitor.get_recommended_quality.return_value = {
            'gpu_enabled': False,
            'parallel_workers': 1
        }
        
        mock_config.is_feature_enabled.return_value = False
        
        chain = FeatureFallbackChain(mock_config, mock_monitor)
        processor = chain.get_processor()
        
        # Should always return a processor (fallback to basic)
        assert processor is not None
        assert hasattr(processor, 'process_frame_batch')
    
    def test_ml_engine_fallback(self, mock_config, mock_monitor):
        """Test ML engine fallback to basic Markov."""
        from core.v2.fallback_system import FeatureFallbackChain
        
        # Disable ML features
        mock_config.is_feature_enabled.return_value = False
        
        chain = FeatureFallbackChain(mock_config, mock_monitor)
        engine = chain.get_ml_engine()
        
        # Should return basic Markov engine
        assert engine is not None
        assert hasattr(engine, 'get_next_state')  # Basic Markov interface
    
    def test_fallback_logging(self, mock_config, mock_monitor):
        """Test fallback event logging."""
        from core.v2.fallback_system import FeatureFallbackChain
        
        chain = FeatureFallbackChain(mock_config, mock_monitor)
        
        # Log a fallback
        chain.log_fallback("GPU Processor", "CPU Processor", "CUDA not available")
        
        # Check fallback was recorded
        assert len(chain.fallback_history) == 1
        
        fallback = chain.fallback_history[0]
        assert fallback['from'] == "GPU Processor"
        assert fallback['to'] == "CPU Processor"
        assert fallback['reason'] == "CUDA not available"
        assert 'timestamp' in fallback
    
    def test_capabilities_reporting(self, mock_config, mock_monitor):
        """Test current capabilities reporting."""
        from core.v2.fallback_system import FeatureFallbackChain
        
        chain = FeatureFallbackChain(mock_config, mock_monitor)
        capabilities = chain.get_current_capabilities()
        
        # Check required capability fields
        required_fields = [
            'processor_type', 'ml_enabled', 'gpu_enabled',
            'real_time_enabled', 'parallel_workers'
        ]
        
        for field in required_fields:
            assert field in capabilities
        
        # Check reasonable values
        assert capabilities['processor_type'] in ['basic', 'parallel_cpu', 'gpu']
        assert isinstance(capabilities['ml_enabled'], bool)
        assert isinstance(capabilities['gpu_enabled'], bool)
        assert isinstance(capabilities['real_time_enabled'], bool)
        assert capabilities['parallel_workers'] >= 1


class TestRealTimeProcessor:
    """Test real-time processing pipeline."""
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self, mock_config, mock_monitor, mock_fallback):
        """Test real-time processor initialization."""
        from advanced.real_time_processor import RealTimeVideoProcessor
        
        processor = RealTimeVideoProcessor(mock_config, mock_monitor, mock_fallback)
        
        assert processor.config == mock_config
        assert processor.monitor == mock_monitor
        assert processor.fallback == mock_fallback
        assert processor.is_streaming is False
    
    @pytest.mark.asyncio
    async def test_stream_buffer(self):
        """Test thread-safe stream buffer."""
        from advanced.real_time_processor import StreamBuffer
        
        buffer = StreamBuffer(max_size=5)
        
        # Test adding frames
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = buffer.put_frame(test_frame, time.time())
        assert result is True
        assert buffer.size() == 1
        
        # Test getting frames
        frame_data = buffer.get_frame()
        assert frame_data is not None
        assert 'frame' in frame_data
        assert 'timestamp' in frame_data
        assert 'frame_id' in frame_data
        
        # Test buffer is empty
        assert buffer.size() == 0
        
        # Test buffer overflow handling
        for i in range(10):  # Add more than max_size
            buffer.put_frame(test_frame, time.time())
        
        assert buffer.size() <= 5  # Should not exceed max_size
    
    @pytest.mark.asyncio
    async def test_connection_type_detection(self, mock_config, mock_monitor, mock_fallback):
        """Test connection type detection."""
        from advanced.real_time_processor import RealTimeVideoProcessor
        
        processor = RealTimeVideoProcessor(mock_config, mock_monitor, mock_fallback)
        
        # Should default to local for safety
        assert processor.connection_type == "local"
    
    @pytest.mark.asyncio
    async def test_auth_token_management(self, mock_config, mock_monitor, mock_fallback):
        """Test authentication token generation and verification."""
        from advanced.real_time_processor import RealTimeVideoProcessor
        
        processor = RealTimeVideoProcessor(mock_config, mock_monitor, mock_fallback)
        
        # Generate token
        token = processor.generate_auth_token()
        assert len(token) >= 32
        assert isinstance(token, str)
        
        # Verify token
        assert processor.verify_token(token) is True
        assert processor.verify_token("invalid_token") is False
    
    def test_performance_metrics(self, mock_config, mock_monitor, mock_fallback):
        """Test stream statistics collection."""
        from advanced.real_time_processor import RealTimeVideoProcessor
        
        processor = RealTimeVideoProcessor(mock_config, mock_monitor, mock_fallback)
        
        stats = processor.get_stream_stats()
        
        # Check required stats fields
        required_fields = [
            'is_streaming', 'target_fps', 'actual_fps',
            'stream_quality', 'buffer_size', 'output_queue_size',
            'error_count', 'recovery_attempts', 'connection_type'
        ]
        
        for field in required_fields:
            assert field in stats
        
        # Check initial values
        assert stats['is_streaming'] is False
        assert stats['error_count'] == 0
        assert stats['recovery_attempts'] == 0


class TestMLTransition:
    """Test ML-enhanced transition detection."""
    
    def test_adaptive_model_initialization(self):
        """Test ML model initialization with different quality modes."""
        from advanced.ml_transition import AdaptiveTransitionModel
        
        # Test CPU mode (should always work)
        model = AdaptiveTransitionModel(quality_mode='cpu', device='cpu')
        
        assert model.quality_mode == 'cpu'
        assert model.device.type == 'cpu'
        assert model.model is not None
        assert model.transform is not None
    
    def test_ml_prediction_with_fallback(self):
        """Test ML prediction with graceful fallback."""
        from advanced.ml_transition import AdaptiveTransitionModel
        
        model = AdaptiveTransitionModel(quality_mode='cpu', device='cpu')
        
        # Create test frames
        frame1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Should not crash and return reasonable scores
        scores = model.predict(frame1, frame2)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3  # [poor, good, excellent]
        assert np.allclose(np.sum(scores), 1.0, atol=0.01)  # Should sum to ~1.0 (softmax)
        assert all(0 <= score <= 1 for score in scores)
    
    def test_ml_caching_system(self):
        """Test ML prediction caching."""
        from advanced.ml_transition import AdaptiveTransitionModel
        
        model = AdaptiveTransitionModel(quality_mode='cpu', device='cpu', cache_size=10)
        
        # Create identical frames
        frame1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # First prediction (cache miss)
        scores1 = model.predict(frame1, frame2)
        initial_misses = model.cache_misses
        
        # Second prediction (should be cache hit)
        scores2 = model.predict(frame1, frame2)
        
        # Should get same results
        np.testing.assert_array_almost_equal(scores1, scores2)
        
        # Cache stats should show hit
        cache_stats = model.get_cache_stats()
        assert cache_stats['cache_hits'] > 0
        assert cache_stats['cache_misses'] == initial_misses  # No new misses
    
    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        from advanced.ml_transition import AdaptiveTransitionModel
        
        model = AdaptiveTransitionModel(quality_mode='cpu', device='cpu')
        
        # Create test frame pairs
        frame_pairs = []
        for i in range(5):
            frame1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            frame2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            frame_pairs.append((frame1, frame2))
        
        # Batch prediction
        results = model.predict_batch(frame_pairs)
        
        assert len(results) == 5
        for scores in results:
            assert isinstance(scores, np.ndarray)
            assert len(scores) == 3
            assert all(0 <= score <= 1 for score in scores)
    
    def test_ml_enhanced_markov_engine(self):
        """Test ML-enhanced Markov engine."""
        from advanced.ml_transition import MLEnhancedMarkovEngine
        
        # Test with ML disabled (should fall back to basic Markov)
        engine = MLEnhancedMarkovEngine(quality_mode='cpu')
        
        # Test basic interface
        assert hasattr(engine, 'predict_transition_quality')
        assert hasattr(engine, 'get_performance_stats')
        
        # Test heuristic fallback
        frame1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Should not crash
        scores = engine.predict_transition_quality(frame1, frame2)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
    
    def test_performance_benchmarking(self):
        """Test ML model performance benchmarking."""
        from advanced.ml_transition import AdaptiveTransitionModel
        
        model = AdaptiveTransitionModel(quality_mode='cpu', device='cpu')
        
        # Run small benchmark
        results = model.benchmark_performance(num_iterations=10)
        
        # Check results structure
        required_fields = [
            'total_time_seconds', 'average_time_ms', 'throughput_fps',
            'quality_mode', 'device', 'iterations'
        ]
        
        for field in required_fields:
            assert field in results
        
        # Check reasonable values
        assert results['total_time_seconds'] > 0
        assert results['average_time_ms'] > 0
        assert results['throughput_fps'] > 0
        assert results['quality_mode'] == 'cpu'
        assert results['iterations'] == 10


class TestPerformanceOptimizations:
    """Test GPU and parallel processing optimizations."""
    
    def test_gpu_processor_initialization(self, mock_config, mock_monitor):
        """Test GPU processor initialization and fallback detection."""
        from advanced.gpu_processor import GPUAcceleratedProcessor
        
        processor = GPUAcceleratedProcessor(mock_config, mock_monitor)
        
        # Should initialize without crashing
        assert processor.config == mock_config
        assert processor.monitor == mock_monitor
        
        # Should have fallback detection
        assert hasattr(processor, 'cuda_available')
        assert hasattr(processor, 'cupy_available')
        assert hasattr(processor, 'pytorch_available')
    
    def test_gpu_frame_processing_fallback(self, mock_config, mock_monitor):
        """Test GPU frame processing with CPU fallback."""
        from advanced.gpu_processor import GPUAcceleratedProcessor
        
        processor = GPUAcceleratedProcessor(mock_config, mock_monitor)
        
        # Test single frame processing
        test_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = processor.process_frame_gpu(test_frame)
        
        # Should return processed frame
        assert isinstance(result, np.ndarray)
        assert result.shape == test_frame.shape
    
    def test_gpu_batch_processing(self, mock_config, mock_monitor):
        """Test GPU batch processing."""
        from advanced.gpu_processor import GPUAcceleratedProcessor
        
        processor = GPUAcceleratedProcessor(mock_config, mock_monitor)
        
        # Test batch processing
        test_frames = np.random.randint(0, 255, (5, 256, 256, 3), dtype=np.uint8)
        results = processor.process_frame_batch_gpu(test_frames)
        
        # Should return processed frames
        assert isinstance(results, np.ndarray)
        assert results.shape == test_frames.shape
    
    def test_parallel_processor_initialization(self):
        """Test parallel CPU processor initialization."""
        from advanced.parallel_processor import ParallelCPUProcessor
        
        processor = ParallelCPUProcessor(num_workers=2)
        
        assert processor.num_workers == 2
        assert processor.executor is not None
        assert len(processor.worker_stats) == 2
    
    def test_parallel_batch_processing(self):
        """Test parallel CPU batch processing."""
        from advanced.parallel_processor import ParallelCPUProcessor
        
        processor = ParallelCPUProcessor(num_workers=2)
        
        # Test batch processing
        test_frames = np.random.randint(0, 255, (10, 128, 128, 3), dtype=np.uint8)
        results = processor.process_frame_batch(test_frames)
        
        # Should return processed frames
        assert isinstance(results, np.ndarray)
        assert results.shape == test_frames.shape
        
        # Cleanup
        processor.cleanup()
    
    def test_adaptive_batch_processing(self):
        """Test adaptive batch processing with target latency."""
        from advanced.parallel_processor import ParallelCPUProcessor
        
        processor = ParallelCPUProcessor(num_workers=2)
        
        # Test adaptive processing
        test_frames = np.random.randint(0, 255, (20, 64, 64, 3), dtype=np.uint8)
        results = processor.process_frame_batch_adaptive(test_frames, target_latency=50.0)
        
        # Should return all frames processed
        assert isinstance(results, np.ndarray)
        assert results.shape == test_frames.shape
        
        # Cleanup
        processor.cleanup()
    
    def test_worker_load_balancing(self):
        """Test worker load balancing statistics."""
        from advanced.parallel_processor import ParallelCPUProcessor
        
        processor = ParallelCPUProcessor(num_workers=3)
        
        # Process some frames to generate statistics
        test_frames = np.random.randint(0, 255, (15, 64, 64, 3), dtype=np.uint8)
        _ = processor.process_frame_batch(test_frames)
        
        # Check load balance stats
        load_balance = processor.get_worker_load_balance()
        
        assert 'workers' in load_balance
        assert 'load_balance_score' in load_balance
        assert 'total_tasks' in load_balance
        
        # Should have stats for all workers
        assert len(load_balance['workers']) == 3
        
        # Load balance score should be between 0 and 1
        assert 0 <= load_balance['load_balance_score'] <= 1
        
        # Cleanup
        processor.cleanup()


class TestIntegrationScenarios:
    """Test integration scenarios and end-to-end functionality."""
    
    def test_v2_system_initialization(self, temp_config_dir):
        """Test complete v2.0 system initialization."""
        from core.v2 import initialize_v2_system
        
        # Should initialize without crashing
        result = initialize_v2_system()
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'version' in result
        assert 'compatibility_mode' in result
        
        # Should work even if some components fail
        if result['success']:
            assert 'system_status' in result
            assert 'recommendations' in result
            assert 'capabilities' in result
    
    def test_system_health_monitoring(self):
        """Test system health monitoring."""
        from core.v2 import get_system_health
        
        health = get_system_health()
        
        assert isinstance(health, dict)
        assert 'overall_health' in health
        assert 'performance_score' in health
        
        # Health should be one of expected values
        assert health['overall_health'] in ['good', 'warning', 'poor', 'error']
        
        # Performance score should be between 0 and 1 (if not error)
        if health['overall_health'] != 'error':
            assert 0 <= health['performance_score'] <= 1
    
    def test_fallback_chain_integration(self, mock_config, mock_monitor):
        """Test complete fallback chain functionality."""
        from core.v2.fallback_system import FeatureFallbackChain
        
        # Test with all features disabled
        mock_config.is_feature_enabled.return_value = False
        mock_monitor.get_recommended_quality.return_value = {
            'gpu_enabled': False,
            'parallel_workers': 1,
            'ml_quality': 'cpu'
        }
        
        chain = FeatureFallbackChain(mock_config, mock_monitor)
        
        # Should get basic components
        processor = chain.get_processor()
        engine = chain.get_ml_engine()
        
        assert processor is not None
        assert engine is not None
        
        # Should record fallbacks
        capabilities = chain.get_current_capabilities()
        assert capabilities['processor_type'] == 'basic'
        assert capabilities['ml_enabled'] is False
    
    def test_memory_pressure_handling(self, mock_config):
        """Test system behavior under memory pressure."""
        from core.v2.resource_monitor import ResourceMonitor
        
        # Mock high memory usage
        mock_config.get.side_effect = lambda key, default=None: {
            'performance.memory_limit_mb': 100,  # Very low limit
            'performance.memory_warning_threshold': 0.5
        }.get(key, default)
        
        monitor = ResourceMonitor(mock_config)
        
        # Should handle emergency cleanup gracefully
        monitor.emergency_cleanup()
        
        # Should still provide recommendations
        recommendations = monitor.get_recommended_quality()
        assert isinstance(recommendations, dict)
    
    @pytest.mark.asyncio
    async def test_websocket_server_graceful_failure(self, mock_config):
        """Test WebSocket server handles initialization failures gracefully."""
        from advanced.websocket_server import WebSocketPreviewServer
        
        # Test with invalid configuration
        mock_config.get.side_effect = lambda key, default=None: {
            'real_time.websocket_port': -1,  # Invalid port
            'real_time.websocket_auth': True
        }.get(key, default)
        
        server = WebSocketPreviewServer(mock_config)
        
        # Should initialize without crashing
        assert server is not None
        
        # Should handle start failure gracefully
        try:
            await server.start()
        except Exception:
            pass  # Expected to fail with invalid port
        
        # Should handle stop gracefully even after failure
        await server.stop()


# Fixtures and test utilities

@pytest.fixture
def temp_config_dir():
    """Create temporary directory for config files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Mock configuration object."""
    config = Mock()
    config.get.return_value = None
    config.is_feature_enabled.return_value = False
    config.get_compatibility_mode.return_value = 'v1'
    return config


@pytest.fixture
def mock_monitor():
    """Mock resource monitor."""
    monitor = Mock()
    monitor.get_system_status.return_value = {
        'cpu_percent': 25.0,
        'memory': {
            'total': 16 * 1024**3,
            'available': 12 * 1024**3,
            'percent': 25.0,
            'used': 4 * 1024**3
        },
        'process': {
            'memory_rss': 100 * 1024**2,
            'memory_percent': 0.6,
            'cpu_percent': 5.0
        },
        'timestamp': time.time()
    }
    
    monitor.get_recommended_quality.return_value = {
        'ml_quality': 'cpu',
        'gpu_enabled': False,
        'parallel_workers': 2,
        'cache_level': 'basic',
        'preview_quality': 'medium',
        'batch_size': 10,
        'real_time_capable': False
    }
    
    monitor.check_memory_usage.return_value = True
    monitor.get_performance_score.return_value = 0.8
    monitor.get_optimization_suggestions.return_value = []
    
    return monitor


@pytest.fixture
def mock_fallback():
    """Mock fallback chain."""
    fallback = Mock()
    
    # Mock processor
    mock_processor = Mock()
    mock_processor.process_frame_batch.return_value = np.array([])
    mock_processor.cleanup.return_value = None
    fallback.get_processor.return_value = mock_processor
    
    # Mock engine
    mock_engine = Mock()
    mock_engine.get_next_state.return_value = 0
    fallback.get_ml_engine.return_value = mock_engine
    fallback.get_basic_markov_engine.return_value = mock_engine
    
    fallback.get_current_capabilities.return_value = {
        'processor_type': 'basic',
        'ml_enabled': False,
        'gpu_enabled': False,
        'real_time_enabled': False,
        'parallel_workers': 1
    }
    
    return fallback


# Performance benchmarks (marked as slow)

@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    def test_ml_performance_benchmark(self):
        """Benchmark ML model performance."""
        from advanced.ml_transition import AdaptiveTransitionModel
        
        model = AdaptiveTransitionModel(quality_mode='cpu', device='cpu')
        results = model.benchmark_performance(num_iterations=50)
        
        # Should complete benchmark
        assert results['throughput_fps'] > 0
        assert results['average_time_ms'] > 0
        
        print(f"ML Benchmark: {results['throughput_fps']:.1f} FPS")
    
    def test_gpu_performance_benchmark(self, mock_config, mock_monitor):
        """Benchmark GPU processor performance."""
        from advanced.gpu_processor import GPUAcceleratedProcessor
        
        processor = GPUAcceleratedProcessor(mock_config, mock_monitor)
        results = processor.benchmark_performance(num_frames=50)
        
        # Should complete benchmark
        assert results['single_frame_fps'] > 0
        
        print(f"GPU Benchmark: {results['single_frame_fps']:.1f} FPS")
        
        processor.cleanup()
    
    def test_parallel_performance_benchmark(self):
        """Benchmark parallel processor performance."""
        from advanced.parallel_processor import ParallelCPUProcessor
        
        processor = ParallelCPUProcessor(num_workers=4)
        results = processor.benchmark_performance(num_frames=100)
        
        # Should complete benchmark
        assert results['adaptive_fps'] > 0
        
        print(f"Parallel Benchmark: {results['adaptive_fps']:.1f} FPS")
        
        processor.cleanup()


# Test markers
pytestmark = pytest.mark.asyncio