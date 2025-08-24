# -*- coding: utf-8 -*-
"""
Configuration Management System for LoopyComfy v2.0

This module provides central configuration management with intelligent fallbacks
and validation for all advanced features.
"""

import yaml
import os
import socket
import secrets
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union


@dataclass
class AdvancedFeaturesConfig:
    """Central configuration for all v2.0 features with validation and fallbacks."""
    
    DEFAULT_CONFIG = """
version: 2.0
compatibility_mode: v1  # Start in v1 mode for safety

advanced_features:
  enabled: false  # Must explicitly enable
  auto_fallback: true  # Automatic degradation
  
real_time:
  enabled: false
  websocket_port: 8765
  websocket_auth: true
  auth_token: "generate_random_token_here"
  max_latency_ms: 
    local: 100
    lan: 200
    internet: 500
  ssl_enabled: false
  ssl_cert_path: null
  ssl_key_path: null
  
ml:
  enabled: false
  model_path: "./models/transition_quality.pth"
  inference_device: "auto"  # auto, cuda, cpu
  quality_mode: "auto"  # full, lite, cpu, auto
  max_inference_ms:
    full: 50
    lite: 30
    cpu: 100
  cache_predictions: true
  continuous_learning: false
  
gpu:
  enabled: false
  device_id: 0
  memory_pool_size_mb: "auto"  # auto or specific size
  cuda_version: "auto"  # auto-detect
  allow_tf32: true
  benchmark_mode: false
  
performance:
  parallel_workers: "auto"  # auto or specific number
  cache_strategy: "intelligent"  # basic, intelligent, off
  memory_limit_mb: null  # null = no limit, or specific value
  memory_warning_threshold: 0.9  # Warn at 90% usage
  
cache:
  redis_enabled: false
  redis_host: "localhost"
  redis_port: 6379
  redis_password: null
  redis_timeout: 5
  memory_cache_size: 100
  disk_cache_enabled: true
  disk_cache_path: "./cache"
  disk_cache_size_gb: 10
  
monitoring:
  enabled: true
  log_level: "INFO"
  performance_tracking: true
  error_reporting: true
  metrics_export: false
  metrics_port: 9090
"""
    
    def __init__(self, config_path="config/advanced_features.yaml"):
        """Initialize configuration with fallback to defaults."""
        self.config_path = config_path
        self.config = self.load_config()
        self.validate_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration with fallback to defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    # Merge with defaults
                    default = yaml.safe_load(self.DEFAULT_CONFIG)
                    return self.deep_merge(default, user_config)
            except Exception as e:
                print(f"Warning: Failed to load config: {e}")
                print("Using default configuration")
        
        # Create default config file
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write(self.DEFAULT_CONFIG)
        
        return yaml.safe_load(self.DEFAULT_CONFIG)
    
    def deep_merge(self, default: Dict, user: Dict) -> Dict:
        """Deep merge user config with defaults."""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(self):
        """Validate configuration and fix issues."""
        # Generate auth token if needed
        if self.config['real_time']['auth_token'] == "generate_random_token_here":
            self.config['real_time']['auth_token'] = secrets.token_urlsafe(32)
            self.save_config()
        
        # Check WebSocket port availability
        if self.config['real_time']['enabled']:
            port = self.config['real_time']['websocket_port']
            if not self.is_port_available(port):
                print(f"Warning: Port {port} not available, using random port")
                self.config['real_time']['websocket_port'] = self.find_free_port()
        
        # Validate ML model path
        if self.config['ml']['enabled']:
            model_path = self.config['ml']['model_path']
            if not os.path.exists(model_path):
                print(f"Warning: ML model not found at {model_path}")
                self.config['ml']['enabled'] = False
        
        # Check GPU availability
        if self.config['gpu']['enabled']:
            try:
                import torch
                if not torch.cuda.is_available():
                    print("Warning: CUDA not available, disabling GPU features")
                    self.config['gpu']['enabled'] = False
            except ImportError:
                print("Warning: PyTorch not available, disabling GPU features")
                self.config['gpu']['enabled'] = False
        
        # Check Redis connection
        if self.config['cache']['redis_enabled']:
            if not self.test_redis_connection():
                print("Warning: Redis not available, disabling Redis cache")
                self.config['cache']['redis_enabled'] = False
        
        # Validate cache directory
        cache_path = self.config['cache']['disk_cache_path']
        try:
            os.makedirs(cache_path, exist_ok=True)
            if not os.access(cache_path, os.W_OK):
                print(f"Warning: Cache directory {cache_path} not writable")
                self.config['cache']['disk_cache_enabled'] = False
        except Exception as e:
            print(f"Warning: Failed to create cache directory: {e}")
            self.config['cache']['disk_cache_enabled'] = False
    
    def is_port_available(self, port: int) -> bool:
        """Check if port is available for binding."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def find_free_port(self) -> int:
        """Find a free port for WebSocket server."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', 0))
                return s.getsockname()[1]
        except OSError:
            # Fallback to a common range
            for port in range(8765, 8800):
                if self.is_port_available(port):
                    return port
            return 8765  # Last resort
    
    def test_redis_connection(self) -> bool:
        """Test Redis availability."""
        try:
            import redis
            r = redis.Redis(
                host=self.config['cache']['redis_host'],
                port=self.config['cache']['redis_port'],
                password=self.config['cache']['redis_password'],
                socket_connect_timeout=self.config['cache']['redis_timeout']
            )
            r.ping()
            return True
        except Exception:
            return False
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save config: {e}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value by dot-separated path."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value by dot-separated path."""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled with proper fallback."""
        # Check global advanced features flag first
        if not self.config.get('advanced_features', {}).get('enabled', False):
            return False
        
        # Check specific feature
        return self.get(f'{feature}.enabled', False)
    
    def get_compatibility_mode(self) -> str:
        """Get compatibility mode (v1, v2, auto)."""
        return self.config.get('compatibility_mode', 'v1')
    
    def enable_v2_features(self):
        """Safely enable v2.0 features with user confirmation."""
        if self.get_compatibility_mode() == 'v1':
            warnings.warn(
                "Enabling v2.0 advanced features. This will change processing behavior. "
                "Ensure you have tested with your workflows first.",
                UserWarning
            )
        
        self.set('compatibility_mode', 'v2')
        self.set('advanced_features.enabled', True)
        self.save_config()
        
        print("v2.0 Advanced features enabled. Restart ComfyUI for full functionality.")


# Global configuration instance
_config_instance: Optional[AdvancedFeaturesConfig] = None


def get_config() -> AdvancedFeaturesConfig:
    """Get global configuration instance (singleton pattern)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AdvancedFeaturesConfig()
    return _config_instance


def reset_config():
    """Reset global configuration instance."""
    global _config_instance
    _config_instance = None


# Convenience functions
def is_v2_enabled() -> bool:
    """Check if v2.0 features are enabled."""
    return get_config().is_feature_enabled('advanced_features')


def is_real_time_enabled() -> bool:
    """Check if real-time features are enabled."""
    return get_config().is_feature_enabled('real_time')


def is_ml_enabled() -> bool:
    """Check if ML features are enabled."""
    return get_config().is_feature_enabled('ml')


def is_gpu_enabled() -> bool:
    """Check if GPU features are enabled."""
    return get_config().is_feature_enabled('gpu')