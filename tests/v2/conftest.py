# -*- coding: utf-8 -*-
"""
Test configuration and fixtures for LoopyComfy v2.0 Advanced Features

This module provides pytest configuration and shared fixtures for testing
all v2.0 components with proper mocking and setup.
"""

import pytest
import asyncio
import numpy as np
import tempfile
import shutil
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Optional


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (may be skipped)"
    )
    config.addinivalue_line(
        "markers", "ml: marks tests that require ML libraries"
    )
    config.addinivalue_line(
        "markers", "websocket: marks tests that use WebSocket functionality"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to benchmark tests
        if "benchmark" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # Add GPU marker to GPU-related tests
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Add ML marker to ML-related tests
        if "ml" in item.nodeid.lower() or "transition" in item.nodeid.lower():
            item.add_marker(pytest.mark.ml)
        
        # Add websocket marker to WebSocket tests
        if "websocket" in item.nodeid.lower() or "real_time" in item.nodeid.lower():
            item.add_marker(pytest.mark.websocket)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# Configuration fixtures

@pytest.fixture
def temp_config_dir():
    """Create temporary directory for configuration files."""
    temp_dir = tempfile.mkdtemp(prefix="loopycomfy_test_config_")
    
    # Create subdirectories
    os.makedirs(os.path.join(temp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "logs"), exist_ok=True)
    
    yield temp_dir
    
    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except (OSError, PermissionError):
        pass  # Ignore cleanup errors


@pytest.fixture
def mock_config():
    """Mock configuration object with reasonable defaults."""
    config = Mock()
    
    # Configuration getters
    config.get.side_effect = lambda key, default=None: {
        'compatibility_mode': 'v1',
        'advanced_features.enabled': False,
        'advanced_features.auto_fallback': True,
        'real_time.enabled': False,
        'real_time.websocket_port': 8765,
        'real_time.websocket_auth': True,
        'real_time.max_latency_ms.local': 100,
        'real_time.max_latency_ms.lan': 200,
        'real_time.max_latency_ms.internet': 500,
        'ml.enabled': False,
        'ml.quality_mode': 'auto',
        'ml.max_inference_ms.full': 50,
        'ml.max_inference_ms.lite': 30,
        'ml.max_inference_ms.cpu': 100,
        'gpu.enabled': False,
        'gpu.memory_pool_size_mb': 'auto',
        'performance.parallel_workers': 'auto',
        'performance.memory_limit_mb': None,
        'performance.memory_warning_threshold': 0.9,
        'cache.redis_enabled': False,
        'cache.disk_cache_enabled': True,
        'cache.disk_cache_path': './cache',
        'monitoring.enabled': True,
        'monitoring.log_level': 'INFO'
    }.get(key, default)
    
    # Feature enablement checks
    config.is_feature_enabled.return_value = False
    config.get_compatibility_mode.return_value = 'v1'
    
    # Configuration path
    config.config_path = "/tmp/test_config.yaml"
    config.config = {
        'compatibility_mode': 'v1',
        'advanced_features': {'enabled': False, 'auto_fallback': True},
        'real_time': {'enabled': False},
        'ml': {'enabled': False},
        'gpu': {'enabled': False}
    }
    
    return config


@pytest.fixture
def mock_resource_monitor():
    """Mock resource monitor with realistic system data."""
    monitor = Mock()
    
    # System status
    monitor.get_system_status.return_value = {
        'cpu_percent': 25.0,
        'memory': {
            'total': 16 * 1024**3,  # 16GB
            'available': 12 * 1024**3,  # 12GB available
            'percent': 25.0,
            'used': 4 * 1024**3  # 4GB used
        },
        'process': {
            'memory_rss': 100 * 1024**2,  # 100MB
            'memory_percent': 0.6,
            'cpu_percent': 5.0
        },
        'gpu': None,  # No GPU by default
        'timestamp': 1640995200.0  # Fixed timestamp for testing
    }
    
    # Quality recommendations
    monitor.get_recommended_quality.return_value = {
        'ml_quality': 'cpu',
        'gpu_enabled': False,
        'parallel_workers': 2,
        'cache_level': 'basic',
        'preview_quality': 'medium',
        'batch_size': 10,
        'real_time_capable': False
    }
    
    # Performance metrics
    monitor.get_performance_score.return_value = 0.8
    monitor.check_memory_usage.return_value = True
    monitor.get_optimization_suggestions.return_value = []
    monitor.should_reduce_quality.return_value = False
    
    # Methods
    monitor.emergency_cleanup.return_value = None
    
    return monitor


@pytest.fixture
def mock_fallback_chain():
    """Mock fallback chain with working components."""
    chain = Mock()
    
    # Mock basic processor
    mock_processor = Mock()
    mock_processor.process_frame_batch.side_effect = lambda frames: frames  # Pass-through
    mock_processor.process_frame_gpu.side_effect = lambda frame: frame
    mock_processor.cleanup.return_value = None
    mock_processor.get_performance_stats.return_value = {
        'cuda_available': False,
        'average_processing_time_ms': 10.0,
        'total_processed_frames': 0
    }
    
    # Mock ML engine
    mock_engine = Mock()
    mock_engine.get_next_state.return_value = 0
    mock_engine.predict_transition_quality.return_value = np.array([0.3, 0.4, 0.3])
    mock_engine.get_performance_stats.return_value = {
        'ml_enabled': False,
        'ml_predictions': 0,
        'fallback_predictions': 0
    }
    
    # Mock WebSocket server
    mock_websocket = Mock()
    mock_websocket.start = Mock()
    mock_websocket.stop = Mock()
    mock_websocket.broadcast_frame = Mock()
    mock_websocket.get_client_count.return_value = 0
    
    # Chain methods
    chain.get_processor.return_value = mock_processor
    chain.get_ml_engine.return_value = mock_engine
    chain.get_basic_markov_engine.return_value = mock_engine
    chain.get_websocket_server.return_value = mock_websocket
    
    chain.get_current_capabilities.return_value = {
        'processor_type': 'basic',
        'ml_enabled': False,
        'gpu_enabled': False,
        'real_time_enabled': False,
        'parallel_workers': 1,
        'ml_quality': None
    }
    
    chain.get_fallback_history.return_value = []
    chain.get_recent_notifications.return_value = []
    chain.log_fallback.return_value = None
    chain.cleanup.return_value = None
    
    return chain


# Test data fixtures

@pytest.fixture
def sample_frames():
    """Generate sample video frames for testing."""
    # Create a batch of test frames (5 frames, 256x256, RGB)
    frames = np.random.randint(0, 255, (5, 256, 256, 3), dtype=np.uint8)
    return frames


@pytest.fixture
def sample_frame():
    """Generate single sample frame for testing."""
    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_video_metadata():
    """Generate sample video metadata for testing."""
    metadata = [
        {
            'path': '/test/video1.mp4',
            'duration': 5.0,
            'fps': 30.0,
            'width': 1920,
            'height': 1080,
            'frame_count': 150,
            'size_mb': 10.5,
            'seamless': True,
            'checksum': 'abc123'
        },
        {
            'path': '/test/video2.mp4',
            'duration': 3.5,
            'fps': 30.0,
            'width': 1920,
            'height': 1080,
            'frame_count': 105,
            'size_mb': 7.2,
            'seamless': True,
            'checksum': 'def456'
        },
        {
            'path': '/test/video3.mp4',
            'duration': 7.2,
            'fps': 30.0,
            'width': 1920,
            'height': 1080,
            'frame_count': 216,
            'size_mb': 15.8,
            'seamless': False,
            'checksum': 'ghi789'
        }
    ]
    return metadata


# Mock fixtures for external dependencies

@pytest.fixture
def mock_torch():
    """Mock PyTorch for testing without GPU requirements."""
    torch_mock = Mock()
    
    # CUDA availability
    torch_mock.cuda.is_available.return_value = False
    torch_mock.cuda.device_count.return_value = 0
    
    # Device mock
    device_mock = Mock()
    device_mock.type = 'cpu'
    torch_mock.device.return_value = device_mock
    
    # Tensor operations
    tensor_mock = Mock()
    tensor_mock.cpu.return_value = tensor_mock
    tensor_mock.numpy.return_value = np.random.rand(3, 224, 224)
    torch_mock.from_numpy.return_value = tensor_mock
    torch_mock.randn.return_value = tensor_mock
    
    with patch.dict('sys.modules', {'torch': torch_mock}):
        yield torch_mock


@pytest.fixture
def mock_cupy():
    """Mock CuPy for testing without CUDA requirements."""
    cupy_mock = Mock()
    
    # CUDA availability
    cupy_mock.cuda.is_available.return_value = False
    
    # Array operations
    array_mock = Mock()
    array_mock.shape = (224, 224, 3)
    cupy_mock.asarray.return_value = array_mock
    cupy_mock.asnumpy.return_value = np.random.rand(224, 224, 3)
    cupy_mock.zeros_like.return_value = array_mock
    
    # Memory pool
    mempool_mock = Mock()
    mempool_mock.free_all_blocks.return_value = None
    cupy_mock.get_default_memory_pool.return_value = mempool_mock
    
    with patch.dict('sys.modules', {'cupy': cupy_mock}):
        yield cupy_mock


@pytest.fixture
def mock_opencv():
    """Mock OpenCV for testing without OpenCV dependency."""
    cv2_mock = Mock()
    
    # Color conversion
    cv2_mock.cvtColor.side_effect = lambda img, code: img[:, :, [2, 1, 0]] if len(img.shape) == 3 else img
    cv2_mock.COLOR_BGR2RGB = 4
    cv2_mock.COLOR_RGB2BGR = 3
    
    # Resize
    cv2_mock.resize.side_effect = lambda img, size, **kwargs: np.random.randint(
        0, 255, (size[1], size[0], img.shape[2] if len(img.shape) == 3 else 1), dtype=np.uint8
    )
    cv2_mock.INTER_LINEAR = 1
    
    # Enhancement
    cv2_mock.convertScaleAbs.side_effect = lambda img, **kwargs: img
    
    with patch.dict('sys.modules', {'cv2': cv2_mock}):
        yield cv2_mock


@pytest.fixture
def mock_websockets():
    """Mock websockets library for testing WebSocket functionality."""
    websockets_mock = Mock()
    
    # Server mock
    server_mock = Mock()
    server_mock.close.return_value = None
    server_mock.wait_closed = Mock()
    
    # WebSocket connection mock
    websocket_mock = Mock()
    websocket_mock.send = Mock()
    websocket_mock.close = Mock()
    websocket_mock.remote_address = ('127.0.0.1', 12345)
    
    # Serve function
    async def mock_serve(*args, **kwargs):
        return server_mock
    
    websockets_mock.serve = mock_serve
    websockets_mock.exceptions = Mock()
    websockets_mock.exceptions.ConnectionClosed = Exception
    websockets_mock.exceptions.ConnectionClosedOK = Exception
    websockets_mock.exceptions.ConnectionClosedError = Exception
    
    with patch.dict('sys.modules', {'websockets': websockets_mock}):
        yield websockets_mock


# Parametrization fixtures

@pytest.fixture(params=['cpu', 'lite', 'full'])
def ml_quality_mode(request):
    """Parametrize ML quality modes."""
    return request.param


@pytest.fixture(params=[1, 2, 4, 8])
def worker_counts(request):
    """Parametrize worker counts for parallel processing."""
    return request.param


@pytest.fixture(params=[1, 5, 10, 20])
def batch_sizes(request):
    """Parametrize batch sizes for testing."""
    return request.param


# Utility fixtures

@pytest.fixture
def performance_timer():
    """Utility for timing test performance."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed()
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0.0
    
    return Timer()


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    import psutil
    import gc
    
    class MemoryMonitor:
        def __init__(self):
            self.initial_memory = None
            self.peak_memory = None
        
        def start(self):
            gc.collect()  # Clean up before monitoring
            self.initial_memory = psutil.Process().memory_info().rss
            self.peak_memory = self.initial_memory
        
        def update(self):
            current_memory = psutil.Process().memory_info().rss
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
        
        def stop(self):
            gc.collect()
            final_memory = psutil.Process().memory_info().rss
            return {
                'initial_mb': self.initial_memory / (1024**2),
                'peak_mb': self.peak_memory / (1024**2),
                'final_mb': final_memory / (1024**2),
                'increase_mb': (final_memory - self.initial_memory) / (1024**2)
            }
    
    return MemoryMonitor()


# Skip conditions

def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    try:
        import torch
        return not torch.cuda.is_available()
    except ImportError:
        return True


def skip_if_no_ml():
    """Skip test if ML libraries are not available."""
    try:
        import torch
        import torchvision
        return False
    except ImportError:
        return True


def skip_if_no_cupy():
    """Skip test if CuPy is not available."""
    try:
        import cupy
        return not cupy.cuda.is_available()
    except ImportError:
        return True


# Pytest markers for skipping
pytestmark_gpu = pytest.mark.skipif(skip_if_no_gpu(), reason="GPU not available")
pytestmark_ml = pytest.mark.skipif(skip_if_no_ml(), reason="ML libraries not available")
pytestmark_cupy = pytest.mark.skipif(skip_if_no_cupy(), reason="CuPy not available")


# Test environment setup

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment with proper isolation."""
    import sys
    import os
    
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        monkeypatch.syspath_prepend(project_root)
    
    # Set environment variables for testing
    monkeypatch.setenv('LOOPYCOMFY_TEST_MODE', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')  # Disable CUDA in tests by default
    
    # Mock external services
    monkeypatch.setenv('REDIS_URL', 'mock://localhost:6379')


# Cleanup fixture

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup resources after each test."""
    yield
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear any remaining GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    # Clear CuPy memory pools
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    except ImportError:
        pass


# Test data validation

def validate_frame(frame):
    """Validate frame format for tests."""
    assert isinstance(frame, np.ndarray)
    assert len(frame.shape) in [2, 3]  # Grayscale or RGB
    if len(frame.shape) == 3:
        assert frame.shape[2] in [3, 4]  # RGB or RGBA
    assert frame.dtype == np.uint8
    assert 0 <= frame.min() <= frame.max() <= 255


def validate_batch(batch):
    """Validate batch of frames."""
    assert isinstance(batch, np.ndarray)
    assert len(batch.shape) == 4  # (N, H, W, C)
    assert batch.shape[0] > 0  # At least one frame
    for frame in batch:
        validate_frame(frame)


# Export fixtures for external use
__all__ = [
    'temp_config_dir',
    'mock_config',
    'mock_resource_monitor',
    'mock_fallback_chain',
    'sample_frames',
    'sample_frame',
    'sample_video_metadata',
    'mock_torch',
    'mock_cupy',
    'mock_opencv',
    'mock_websockets',
    'ml_quality_mode',
    'worker_counts',
    'batch_sizes',
    'performance_timer',
    'memory_monitor',
    'validate_frame',
    'validate_batch'
]