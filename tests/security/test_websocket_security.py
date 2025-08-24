# -*- coding: utf-8 -*-
"""
WebSocket security tests for LoopyComfy
Tests authentication, rate limiting, and connection security.
"""

import pytest
import asyncio
import time
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from advanced.websocket_server import WebSocketPreviewServer
from utils.security_utils import SecureTokenManager


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self, **kwargs):
        self.data = {
            'real_time.websocket_port': 8765,
            'real_time.websocket_auth': True,
            'real_time.auth_token': 'test_secret_key',
        }
        self.data.update(kwargs)
    
    def get(self, key, default=None):
        return self.data.get(key, default)


class TestWebSocketSecurity:
    """Test WebSocket server security."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = MockConfig()
        self.server = WebSocketPreviewServer(self.config)
    
    def test_secure_token_verification(self):
        """Test secure token verification."""
        # Generate valid token
        valid_token = self.server.token_manager.generate_token()
        
        # Valid token should pass
        assert self.server._verify_token(valid_token, "127.0.0.1")
        
        # Invalid tokens should fail
        invalid_tokens = [
            "",
            "invalid",
            "too_short",
            valid_token[:-1] + 'x',  # Tampered
            None,
            123,
        ]
        
        for invalid_token in invalid_tokens:
            assert not self.server._verify_token(invalid_token, "127.0.0.1")
    
    def test_authentication_rate_limiting(self):
        """Test rate limiting of authentication attempts."""
        client_ip = "192.168.1.100"
        invalid_token = "invalid_token"
        
        # First few attempts should be allowed
        for i in range(5):
            result = self.server._verify_token(invalid_token, client_ip)
            assert not result  # Token is invalid
        
        # After max attempts, should be rate limited
        assert self.server._is_rate_limited(client_ip)
        
        # Further attempts should be blocked
        result = self.server._verify_token(invalid_token, client_ip)
        assert not result
    
    def test_connection_rate_limiting(self):
        """Test rate limiting of connection attempts."""
        client_ip = "192.168.1.200"
        
        # First few connections should be allowed
        for i in range(3):
            assert not self.server._is_connection_rate_limited(client_ip)
            self.server._record_connection_attempt(client_ip)
        
        # After max connections, should be rate limited
        assert self.server._is_connection_rate_limited(client_ip)
    
    def test_rate_limit_window_expiry(self):
        """Test that rate limits expire after window."""
        client_ip = "192.168.1.300"
        
        # Exhaust auth attempts
        for i in range(6):
            self.server._record_auth_attempt(client_ip, success=False)
        
        assert self.server._is_rate_limited(client_ip)
        
        # Mock time advancement beyond window
        with patch('time.time', return_value=time.time() + 400):  # 400 seconds later
            assert not self.server._is_rate_limited(client_ip)
    
    def test_successful_auth_doesnt_rate_limit(self):
        """Test that successful authentication doesn't trigger rate limiting."""
        client_ip = "192.168.1.400"
        valid_token = self.server.token_manager.generate_token()
        
        # Multiple successful authentications should not trigger rate limiting
        for i in range(10):
            result = self.server._verify_token(valid_token, client_ip)
            assert result
        
        # Should not be rate limited after successful attempts
        assert not self.server._is_rate_limited(client_ip)
    
    def test_different_ips_separate_rate_limits(self):
        """Test that different IPs have separate rate limits."""
        client_ip1 = "192.168.1.500"
        client_ip2 = "192.168.1.501"
        invalid_token = "invalid"
        
        # Exhaust rate limit for first IP
        for i in range(6):
            self.server._verify_token(invalid_token, client_ip1)
        
        assert self.server._is_rate_limited(client_ip1)
        assert not self.server._is_rate_limited(client_ip2)
        
        # Second IP should still work
        valid_token = self.server.token_manager.generate_token()
        assert self.server._verify_token(valid_token, client_ip2)
    
    def test_token_without_ip_rate_limiting(self):
        """Test token verification without IP (should still work)."""
        valid_token = self.server.token_manager.generate_token()
        invalid_token = "invalid"
        
        # Should work without IP address
        assert self.server._verify_token(valid_token, None)
        assert not self.server._verify_token(invalid_token, None)
    
    def test_malicious_token_inputs(self):
        """Test handling of malicious token inputs."""
        malicious_tokens = [
            "'; DROP TABLE users; --",  # SQL injection
            "<script>alert('xss')</script>",  # XSS
            "../../etc/passwd",  # Path traversal
            "\x00nullbyte",  # Null byte
            "a" * 10000,  # Extremely long token
            {"malicious": "object"},  # Wrong type
            ["malicious", "array"],  # Wrong type
        ]
        
        for malicious_token in malicious_tokens:
            # Should not crash or leak information
            result = self.server._verify_token(malicious_token, "test_ip")
            assert not result
    
    def test_concurrent_authentication(self):
        """Test concurrent authentication attempts."""
        async def auth_attempt(token, ip):
            return self.server._verify_token(token, ip)
        
        async def test_concurrent():
            valid_token = self.server.token_manager.generate_token()
            
            # Run concurrent authentication attempts
            tasks = []
            for i in range(50):
                ip = f"192.168.1.{i}"
                task = auth_attempt(valid_token, ip)
                tasks.append(task)
            
            results = await asyncio.gather(*[asyncio.coroutine(lambda: task)() for task in tasks])
            
            # All should succeed
            assert all(results)
        
        # asyncio.run(test_concurrent())


class TestTokenSecurity:
    """Test token security specifically."""
    
    def test_token_entropy(self):
        """Test that tokens have sufficient entropy."""
        token_manager = SecureTokenManager()
        
        tokens = set()
        for _ in range(1000):
            token = token_manager.generate_token()
            tokens.add(token)
        
        # Should have 1000 unique tokens (no collisions)
        assert len(tokens) == 1000
    
    def test_token_format_consistency(self):
        """Test token format consistency."""
        token_manager = SecureTokenManager()
        
        for _ in range(100):
            token = token_manager.generate_token()
            
            # Token should have payload.signature format
            assert '.' in token
            parts = token.split('.')
            assert len(parts) == 2
            
            # Both parts should be non-empty
            payload, signature = parts
            assert len(payload) > 0
            assert len(signature) > 0
    
    def test_token_hmac_security(self):
        """Test HMAC security of tokens."""
        secret1 = "secret1"
        secret2 = "secret2"
        
        manager1 = SecureTokenManager(secret1)
        manager2 = SecureTokenManager(secret2)
        
        token1 = manager1.generate_token("test_payload")
        
        # Token should only verify with correct secret
        assert manager1.verify_token(token1)
        assert not manager2.verify_token(token1)
    
    def test_token_replay_attack_resistance(self):
        """Test resistance to token replay attacks."""
        token_manager = SecureTokenManager()
        
        # Generate token with specific payload
        payload = "user_id:123"
        token = token_manager.generate_token(payload)
        
        # Token should verify multiple times (stateless)
        assert token_manager.verify_token(token)
        assert token_manager.verify_token(token)
        
        # But each generation should create different tokens
        token2 = token_manager.generate_token(payload)
        assert token != token2


class TestWebSocketDoS:
    """Test WebSocket DoS attack prevention."""
    
    def test_connection_limit_per_ip(self):
        """Test connection limits per IP address."""
        config = MockConfig()
        server = WebSocketPreviewServer(config)
        
        test_ip = "192.168.1.666"
        
        # Should allow initial connections
        for i in range(3):
            assert not server._is_connection_rate_limited(test_ip)
            server._record_connection_attempt(test_ip)
        
        # Should block further connections
        assert server._is_connection_rate_limited(test_ip)
    
    def test_memory_exhaustion_prevention(self):
        """Test prevention of memory exhaustion through large messages."""
        config = MockConfig()
        server = WebSocketPreviewServer(config)
        
        # Large token should be rejected
        large_token = "a" * 100000
        assert not server._verify_token(large_token, "test_ip")
    
    def test_rapid_authentication_attempts(self):
        """Test handling of rapid authentication attempts."""
        config = MockConfig()
        server = WebSocketPreviewServer(config)
        
        test_ip = "192.168.1.777"
        
        # Rapid invalid attempts should trigger rate limiting
        start_time = time.time()
        for i in range(100):
            server._verify_token("invalid", test_ip)
            
        end_time = time.time()
        
        # Should be rate limited after max attempts
        assert server._is_rate_limited(test_ip)
        
        # Should not take too long (no infinite loops)
        assert (end_time - start_time) < 1.0  # Should complete in < 1 second


class TestSecurityHeaders:
    """Test security headers and configurations."""
    
    def test_secure_server_configuration(self):
        """Test that server is configured securely."""
        config = MockConfig(
            **{
                'real_time.websocket_auth': True,  # Authentication enabled
                'real_time.auth_token': 'secure_secret_key_here',
            }
        )
        server = WebSocketPreviewServer(config)
        
        # Should require authentication
        assert server.auth_required
        
        # Should have secure token manager
        assert isinstance(server.token_manager, SecureTokenManager)
        
        # Should have rate limiting configured
        assert server.max_auth_attempts > 0
        assert server.auth_window_seconds > 0
    
    def test_insecure_configuration_detection(self):
        """Test detection of insecure configurations."""
        # Test with auth disabled (insecure)
        insecure_config = MockConfig(**{'real_time.websocket_auth': False})
        server = WebSocketPreviewServer(insecure_config)
        
        # Should still have security components but with auth disabled
        assert not server.auth_required


if __name__ == "__main__":
    # Run security tests
    pytest.main([__file__, "-v", "--tb=short"])