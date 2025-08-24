# -*- coding: utf-8 -*-
"""
LoopyComfy v2.0 Advanced Features Test Suite

This package contains comprehensive tests for all v2.0 advanced features
including configuration management, resource monitoring, fallback systems,
real-time processing, ML-enhanced transitions, and performance optimizations.

Test Categories:
- Configuration Management: Config loading, validation, feature flags
- Resource Monitoring: System status, adaptive quality, memory management
- Fallback Systems: Intelligent degradation, component fallbacks
- Real-Time Processing: WebSocket streaming, adaptive latency
- ML Enhancements: Transition detection, model adaptation
- Performance Optimizations: GPU acceleration, parallel processing
- Integration Tests: End-to-end workflows, system health

Usage:
    # Run all v2.0 tests
    pytest tests/v2/
    
    # Run specific test categories
    pytest tests/v2/ -m "not slow"  # Skip slow benchmarks
    pytest tests/v2/ -m integration  # Only integration tests
    pytest tests/v2/ -m gpu  # Only GPU-related tests
    
    # Run with coverage
    pytest tests/v2/ --cov=core.v2 --cov=advanced
"""

__version__ = "2.0.0"
__author__ = "LoopyComfy Team"

# Test markers for organization
TEST_MARKERS = {
    'slow': 'Performance benchmarks and long-running tests',
    'integration': 'End-to-end integration tests',
    'gpu': 'Tests requiring GPU hardware',
    'ml': 'Tests requiring ML libraries (PyTorch, etc)',
    'websocket': 'Tests using WebSocket functionality'
}

# Test configuration
TEST_CONFIG = {
    'timeout': 30,  # Default test timeout in seconds
    'memory_limit_mb': 1000,  # Memory limit for test processes
    'gpu_memory_limit_mb': 512,  # GPU memory limit for tests
    'parallel_workers': 2,  # Default worker count for parallel tests
}

# Export test utilities
from .conftest import (
    validate_frame,
    validate_batch,
    skip_if_no_gpu,
    skip_if_no_ml,
    skip_if_no_cupy
)

__all__ = [
    'TEST_MARKERS',
    'TEST_CONFIG',
    'validate_frame',
    'validate_batch',
    'skip_if_no_gpu',
    'skip_if_no_ml',
    'skip_if_no_cupy'
]