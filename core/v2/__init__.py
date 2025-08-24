# -*- coding: utf-8 -*-
"""
LoopyComfy v2.0 Advanced Features Core Module

This module provides the foundation for all v2.0 advanced features including
configuration management, resource monitoring, and intelligent fallback systems.
"""

from .config_manager import (
    AdvancedFeaturesConfig, 
    get_config, 
    reset_config,
    is_v2_enabled,
    is_real_time_enabled,
    is_ml_enabled,
    is_gpu_enabled
)

from .resource_monitor import ResourceMonitor

from .fallback_system import FeatureFallbackChain

__version__ = "2.0.0"
__author__ = "LoopyComfy Team"

# Export main classes
__all__ = [
    'AdvancedFeaturesConfig',
    'ResourceMonitor', 
    'FeatureFallbackChain',
    'get_config',
    'reset_config',
    'is_v2_enabled',
    'is_real_time_enabled',
    'is_ml_enabled',
    'is_gpu_enabled'
]


def initialize_v2_system():
    """
    Initialize v2.0 system with proper checks and fallbacks.
    
    Returns:
        dict: System initialization status and capabilities
    """
    try:
        # Load configuration
        config = get_config()
        
        # Initialize resource monitor
        monitor = ResourceMonitor(config)
        
        # Initialize fallback system
        fallback_chain = FeatureFallbackChain(config, monitor)
        
        # Get system status and recommendations
        system_status = monitor.get_system_status()
        recommendations = monitor.get_recommended_quality()
        capabilities = fallback_chain.get_current_capabilities()
        
        initialization_result = {
            'success': True,
            'version': __version__,
            'compatibility_mode': config.get_compatibility_mode(),
            'v2_features_enabled': is_v2_enabled(),
            'system_status': system_status,
            'recommendations': recommendations,
            'capabilities': capabilities,
            'fallback_history': fallback_chain.get_fallback_history(),
            'config_path': config.config_path
        }
        
        # Log initialization
        print(f"LoopyComfy v{__version__} initialized successfully")
        print(f"Compatibility mode: {config.get_compatibility_mode()}")
        print(f"Advanced features: {'Enabled' if is_v2_enabled() else 'Disabled'}")
        
        if recommendations.get('real_time_capable'):
            print("System capable of real-time processing")
        
        return initialization_result
        
    except Exception as e:
        print(f"LoopyComfy v2.0 initialization failed: {e}")
        print("Falling back to v1.x compatibility mode")
        
        return {
            'success': False,
            'version': __version__,
            'error': str(e),
            'compatibility_mode': 'v1',
            'v2_features_enabled': False,
            'fallback_reason': 'Initialization failed'
        }


def get_system_health():
    """
    Get current system health and performance metrics.
    
    Returns:
        dict: Health status and metrics
    """
    try:
        config = get_config()
        monitor = ResourceMonitor(config)
        
        status = monitor.get_system_status()
        score = monitor.get_performance_score()
        suggestions = monitor.get_optimization_suggestions()
        
        return {
            'overall_health': 'good' if score > 0.7 else 'warning' if score > 0.3 else 'poor',
            'performance_score': score,
            'system_status': status,
            'optimization_suggestions': suggestions,
            'memory_usage_ok': monitor.check_memory_usage(),
            'timestamp': status['timestamp']
        }
        
    except Exception as e:
        return {
            'overall_health': 'error',
            'error': str(e),
            'timestamp': time.time()
        }
        

def cleanup_v2_system():
    """Clean up v2.0 system resources."""
    try:
        config = get_config()
        monitor = ResourceMonitor(config)
        fallback_chain = FeatureFallbackChain(config, monitor)
        
        # Cleanup fallback system
        fallback_chain.cleanup()
        
        # Emergency cleanup if needed
        if not monitor.check_memory_usage():
            monitor.emergency_cleanup()
        
        print("LoopyComfy v2.0 system cleaned up successfully")
        
    except Exception as e:
        print(f"Cleanup error: {e}")