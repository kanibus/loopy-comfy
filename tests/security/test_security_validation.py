# -*- coding: utf-8 -*-
"""
Security validation tests for LoopyComfy
Tests path validation, input sanitization, and authentication security.
"""

import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from utils.security_utils import (
    PathValidator, InputValidator, SecureTokenManager,
    ResourceLimiter, SecurityError
)


class TestPathValidator:
    """Test path validation security."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = PathValidator()
        # Create temporary test directory
        self.test_dir = tempfile.mkdtemp()
        self.allowed_dirs = [self.test_dir]
        self.secure_validator = PathValidator(self.allowed_dirs)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_path_traversal_attacks(self):
        """Test path traversal attack prevention."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
            "assets/../../../sensitive",
            "/etc/passwd",
            "C:\\Windows\\System32",
            "~/../../etc/passwd",
            "$HOME/../../../etc/passwd",
            "\x00../../../etc/passwd",  # Null byte injection
            "assets\x00/../../../etc/passwd",
        ]
        
        for malicious_path in malicious_paths:
            with pytest.raises(SecurityError):
                self.secure_validator.validate_directory_path(malicious_path)
    
    def test_valid_paths(self):
        """Test that valid paths are accepted."""
        # Create subdirectory in test directory
        subdir = os.path.join(self.test_dir, "videos")
        os.makedirs(subdir, exist_ok=True)
        
        valid_paths = [
            self.test_dir,
            subdir,
            os.path.join(self.test_dir, "assets"),
        ]
        
        for valid_path in valid_paths:
            if not os.path.exists(valid_path):
                os.makedirs(valid_path, exist_ok=True)
            
            # Should not raise exception
            result = self.secure_validator.validate_directory_path(valid_path)
            assert os.path.isabs(result)
    
    def test_file_pattern_injection(self):
        """Test file pattern injection prevention."""
        malicious_patterns = [
            "*.mp4; rm -rf /",
            "*.mp4 && malicious_command",
            "*.mp4 | evil_script",
            "*.mp4$(rm -rf /)",
            "*.mp4`malicious`",
            "*.mp4\x00; rm -rf /",  # Null byte injection
            "*.mp4{dangerous,payload}",
            "*.mp4<script>alert('xss')</script>",
            "*.mp4;echo 'pwned'",
            "../../../*.mp4",
            "*/../../*.mp4",
        ]
        
        for malicious_pattern in malicious_patterns:
            with pytest.raises(SecurityError):
                self.validator.validate_file_pattern(malicious_pattern)
    
    def test_valid_file_patterns(self):
        """Test that valid file patterns are accepted."""
        valid_patterns = [
            "*.mp4",
            "*.avi",
            "avatar_*.mov",
            "test.mp4",
            "video_[0-9].mp4",
            "*.webm",
            "clip*.mkv",
        ]
        
        for valid_pattern in valid_patterns:
            # Should not raise exception
            result = self.validator.validate_file_pattern(valid_pattern)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_path_length_limits(self):
        """Test path length validation."""
        # Test extremely long path
        long_path = "a" * 1000
        
        with pytest.raises(SecurityError):
            self.validator.validate_directory_path(long_path)
    
    def test_suspicious_characters(self):
        """Test detection of suspicious characters in paths."""
        suspicious_paths = [
            "path/with\x00nullbyte",
            "path/with\x01control",
            "path<script>alert(1)</script>",
            "path|with|pipes",
            "path>with>redirects",
            "path${VAR}expansion",
        ]
        
        for suspicious_path in suspicious_paths:
            with pytest.raises(SecurityError):
                self.secure_validator.validate_directory_path(suspicious_path)


class TestInputValidator:
    """Test input validation security."""
    
    def setup_method(self):
        """Set up test environment."""
        self.validator = InputValidator()
    
    def test_string_input_validation(self):
        """Test string input validation."""
        # Test length limits
        with pytest.raises(SecurityError):
            self.validator.validate_string_input("a" * 2000, max_length=1000)
        
        # Test control characters
        with pytest.raises(SecurityError):
            self.validator.validate_string_input("test\x00string")
        
        # Test valid string
        result = self.validator.validate_string_input("valid string", max_length=100)
        assert result == "valid string"
    
    def test_numeric_input_validation(self):
        """Test numeric input validation."""
        # Test range validation
        with pytest.raises(SecurityError):
            self.validator.validate_numeric_input(150, min_val=0, max_val=100)
        
        with pytest.raises(SecurityError):
            self.validator.validate_numeric_input(-50, min_val=0, max_val=100)
        
        # Test string to numeric conversion
        result = self.validator.validate_numeric_input("42.5", min_val=0, max_val=100)
        assert result == 42.5
        
        # Test malicious string input
        with pytest.raises(SecurityError):
            self.validator.validate_numeric_input("42; rm -rf /")
    
    def test_sql_injection_patterns(self):
        """Test detection of SQL injection patterns."""
        sql_injection_strings = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--",
            "'; EXEC xp_cmdshell('format c:')--",
        ]
        
        for injection_string in sql_injection_strings:
            # Should sanitize or reject malicious SQL
            result = self.validator.validate_string_input(injection_string, max_length=1000)
            # Control characters should be removed
            assert '\x00' not in result


class TestSecureTokenManager:
    """Test secure token management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.token_manager = SecureTokenManager()
    
    def test_token_generation(self):
        """Test secure token generation."""
        token1 = self.token_manager.generate_token()
        token2 = self.token_manager.generate_token()
        
        # Tokens should be different
        assert token1 != token2
        
        # Tokens should be verifiable
        assert self.token_manager.verify_token(token1)
        assert self.token_manager.verify_token(token2)
    
    def test_token_tampering_detection(self):
        """Test detection of token tampering."""
        token = self.token_manager.generate_token()
        
        # Tamper with token
        tampered_token = token[:-1] + 'x'
        
        # Should reject tampered token
        assert not self.token_manager.verify_token(tampered_token)
    
    def test_invalid_token_formats(self):
        """Test rejection of invalid token formats."""
        invalid_tokens = [
            "",
            "invalid",
            "no.signature",
            ".nosignature",
            "payload.",
            None,
            123,
            "a" * 10000,  # Extremely long token
        ]
        
        for invalid_token in invalid_tokens:
            assert not self.token_manager.verify_token(invalid_token)
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        import time
        
        valid_token = self.token_manager.generate_token()
        invalid_token = "invalid_token_format"
        
        # Measure verification times
        times_valid = []
        times_invalid = []
        
        for _ in range(100):
            start = time.perf_counter()
            self.token_manager.verify_token(valid_token)
            times_valid.append(time.perf_counter() - start)
            
            start = time.perf_counter()
            self.token_manager.verify_token(invalid_token)
            times_invalid.append(time.perf_counter() - start)
        
        # Timing should be relatively consistent (no obvious timing attacks)
        avg_valid = sum(times_valid) / len(times_valid)
        avg_invalid = sum(times_invalid) / len(times_invalid)
        
        # Times should be in similar range (within order of magnitude)
        assert abs(avg_valid - avg_invalid) < max(avg_valid, avg_invalid)


class TestResourceLimiter:
    """Test resource usage limiting."""
    
    def setup_method(self):
        """Set up test environment."""
        self.limiter = ResourceLimiter(max_memory_mb=100, max_files=10)
    
    def test_file_limit_enforcement(self):
        """Test file processing limit enforcement."""
        # Should start within limits
        assert self.limiter.check_file_limit()
        
        # Exceed file limit
        for i in range(15):
            self.limiter.increment_file_count()
        
        # Should exceed limit
        assert not self.limiter.check_file_limit()
    
    @patch('psutil.Process')
    def test_memory_limit_enforcement(self, mock_process):
        """Test memory limit enforcement."""
        # Mock memory usage within limits
        mock_process.return_value.memory_info.return_value.rss = 50 * 1024 * 1024  # 50MB
        
        assert self.limiter.check_memory_usage()
        
        # Mock memory usage exceeding limits
        mock_process.return_value.memory_info.return_value.rss = 200 * 1024 * 1024  # 200MB
        
        assert not self.limiter.check_memory_usage()


class TestSecurityIntegration:
    """Integration tests for security components."""
    
    def test_combined_path_and_input_validation(self):
        """Test combined security validation."""
        path_validator = PathValidator()
        input_validator = InputValidator()
        
        # Test malicious combined input
        malicious_inputs = [
            ("../../../etc/passwd", "*.mp4; rm -rf /"),
            ("${HOME}/../../sensitive", "malicious_pattern"),
            ("\x00nullbyte/path", "pattern\x00injection"),
        ]
        
        for malicious_path, malicious_pattern in malicious_inputs:
            with pytest.raises(SecurityError):
                path_validator.validate_directory_path(malicious_path)
            
            with pytest.raises(SecurityError):
                path_validator.validate_file_pattern(malicious_pattern)
    
    def test_fuzzing_style_inputs(self):
        """Test with fuzzing-style random inputs."""
        import random
        import string
        
        path_validator = PathValidator()
        
        # Generate random potentially malicious inputs
        for _ in range(100):
            # Random string with potential injection characters
            chars = string.ascii_letters + string.digits + "../\\;|&$`()"
            random_input = ''.join(random.choices(chars, k=random.randint(1, 200)))
            
            try:
                result = path_validator.validate_directory_path(random_input)
                # If validation passes, result should be safe
                assert not any(dangerous in result for dangerous in ['..', ';', '|', '&'])
            except SecurityError:
                # Expected for most random inputs
                pass
            except Exception as e:
                # Should not crash with unexpected exceptions
                pytest.fail(f"Unexpected exception for input '{random_input}': {e}")
    
    def test_dos_attack_prevention(self):
        """Test prevention of DoS attacks through resource exhaustion."""
        limiter = ResourceLimiter(max_memory_mb=1, max_files=5)
        
        # Should prevent excessive file processing
        for i in range(10):
            limiter.increment_file_count()
        
        assert not limiter.check_file_limit()
    
    def test_unicode_security(self):
        """Test security with Unicode inputs."""
        path_validator = PathValidator()
        
        # Test Unicode normalization attacks
        unicode_attacks = [
            "ℱi︎le",  # Unicode confusables
            "․․/secret",  # One-dot leader characters
            "＜script＞alert(1)＜/script＞",  # Fullwidth characters
            "file\u202e.txt",  # Right-to-Left Override
            "\uFEFFfile.txt",  # Zero Width No-Break Space
        ]
        
        for unicode_attack in unicode_attacks:
            try:
                result = path_validator.validate_directory_path(unicode_attack)
                # Should handle Unicode safely
                assert isinstance(result, str)
            except SecurityError:
                # Expected for suspicious Unicode patterns
                pass


if __name__ == "__main__":
    # Run security tests
    pytest.main([__file__, "-v", "--tb=short"])