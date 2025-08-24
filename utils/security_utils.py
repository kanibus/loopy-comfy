# -*- coding: utf-8 -*-
"""
Security utilities for LoopyComfy
Provides secure path validation, input sanitization, and cryptographic utilities.
"""

import os
import re
import hashlib
import secrets
import hmac
import logging
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import unquote


# Configure security logger
security_logger = logging.getLogger('loopycomfy.security')
security_logger.setLevel(logging.INFO)


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class PathValidator:
    """Secure path validation and sanitization."""
    
    def __init__(self, allowed_base_dirs: Optional[List[str]] = None):
        """
        Initialize path validator with allowed base directories.
        
        Args:
            allowed_base_dirs: List of allowed base directories. 
                             Defaults to ComfyUI safe directories.
        """
        if allowed_base_dirs is None:
            # Default safe directories for ComfyUI
            self.allowed_base_dirs = [
                "./assets/",
                "./input/", 
                "./ComfyUI/input/",
                "./ComfyUI/output/",
                "./ComfyUI/custom_nodes/loopy-comfy/assets/",
                "./temp/"
            ]
        else:
            self.allowed_base_dirs = allowed_base_dirs
            
        # Resolve all allowed paths to absolute paths
        self.resolved_allowed_dirs = []
        for base_dir in self.allowed_base_dirs:
            try:
                resolved = os.path.realpath(os.path.abspath(base_dir))
                self.resolved_allowed_dirs.append(resolved)
                # Create directory if it doesn't exist (for temp dirs)
                os.makedirs(resolved, exist_ok=True)
            except (OSError, ValueError) as e:
                security_logger.warning(f"Could not resolve allowed directory {base_dir}: {e}")
    
    def validate_directory_path(self, directory_path: str) -> str:
        """
        Validate and sanitize a directory path.
        
        Args:
            directory_path: Path to validate
            
        Returns:
            Validated absolute path
            
        Raises:
            SecurityError: If path is not safe
        """
        if not directory_path or not isinstance(directory_path, str):
            raise SecurityError("Invalid directory path: empty or not string")
        
        # Decode URL encoding and remove null bytes
        try:
            clean_path = unquote(directory_path.strip())
            clean_path = clean_path.replace('\x00', '')  # Remove null bytes
        except Exception:
            raise SecurityError("Invalid path encoding")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'\.{2,}',  # Multiple dots
            r'[<>:"|?*]',  # Invalid filename chars
            r'[\x00-\x1f]',  # Control characters
            r'\\{2,}',  # Multiple backslashes
            r'/\.',  # Hidden directories
            r'~/',  # Home directory references
            r'\$\{',  # Environment variable expansion
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, clean_path):
                security_logger.warning(f"Suspicious pattern in path: {clean_path}")
                raise SecurityError(f"Potentially unsafe path pattern detected")
        
        # Resolve to absolute path
        try:
            abs_path = os.path.realpath(os.path.abspath(clean_path))
        except (OSError, ValueError):
            raise SecurityError("Cannot resolve path to absolute form")
        
        # Check if path is within allowed directories
        path_allowed = False
        for allowed_dir in self.resolved_allowed_dirs:
            if abs_path.startswith(allowed_dir):
                path_allowed = True
                break
        
        if not path_allowed:
            security_logger.error(f"Path access denied: {abs_path}")
            raise SecurityError("Directory access not permitted")
        
        # Additional checks for Windows
        if os.name == 'nt':
            if len(abs_path) > 260:  # Windows MAX_PATH
                raise SecurityError("Path too long for Windows filesystem")
            
        security_logger.info(f"Path validated: {abs_path}")
        return abs_path
    
    def validate_file_pattern(self, file_pattern: str) -> str:
        """
        Validate and sanitize a file pattern for glob operations.
        
        Args:
            file_pattern: File pattern to validate (e.g., "*.mp4")
            
        Returns:
            Validated file pattern
            
        Raises:
            SecurityError: If pattern is not safe
        """
        if not file_pattern or not isinstance(file_pattern, str):
            raise SecurityError("Invalid file pattern: empty or not string")
        
        # Remove null bytes and decode
        clean_pattern = file_pattern.replace('\x00', '').strip()
        
        # Whitelist approach - only allow safe characters for glob patterns
        allowed_chars = re.compile(r'^[a-zA-Z0-9._*?[\]-]+$')
        if not allowed_chars.match(clean_pattern):
            raise SecurityError("File pattern contains unsafe characters")
        
        # Prevent directory traversal in patterns
        if '..' in clean_pattern or '/' in clean_pattern or '\\' in clean_pattern:
            raise SecurityError("Path separators not allowed in file pattern")
        
        # Limit pattern length to prevent ReDoS attacks
        if len(clean_pattern) > 100:
            raise SecurityError("File pattern too long")
        
        # Ensure pattern looks like a valid glob
        if not any(char in clean_pattern for char in ['*', '?', '[']):
            # If no glob chars, assume it's a file extension
            if '.' not in clean_pattern:
                clean_pattern = '*.' + clean_pattern
            elif not clean_pattern.startswith('*.'):
                clean_pattern = '*' + clean_pattern
        
        security_logger.info(f"File pattern validated: {clean_pattern}")
        return clean_pattern


class InputValidator:
    """General input validation utilities."""
    
    @staticmethod
    def validate_string_input(value: str, max_length: int = 1000, 
                            allowed_chars: Optional[str] = None) -> str:
        """
        Validate and sanitize string input.
        
        Args:
            value: String to validate
            max_length: Maximum allowed length
            allowed_chars: Regex pattern for allowed characters
            
        Returns:
            Validated string
            
        Raises:
            SecurityError: If string is not safe
        """
        if not isinstance(value, str):
            raise SecurityError("Input must be string")
        
        # Remove null bytes and control characters
        clean_value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)
        
        # Check length
        if len(clean_value) > max_length:
            raise SecurityError(f"Input too long: {len(clean_value)} > {max_length}")
        
        # Check allowed characters if specified
        if allowed_chars and not re.match(allowed_chars, clean_value):
            raise SecurityError("Input contains disallowed characters")
        
        return clean_value
    
    @staticmethod
    def validate_numeric_input(value: Union[int, float, str], 
                             min_val: Optional[float] = None,
                             max_val: Optional[float] = None) -> float:
        """
        Validate numeric input.
        
        Args:
            value: Numeric value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Validated numeric value
            
        Raises:
            SecurityError: If value is not safe
        """
        try:
            if isinstance(value, str):
                # Remove any non-numeric characters except decimal point and minus
                clean_value = re.sub(r'[^\d.-]', '', value)
                numeric_value = float(clean_value)
            else:
                numeric_value = float(value)
        except (ValueError, TypeError):
            raise SecurityError("Invalid numeric input")
        
        if min_val is not None and numeric_value < min_val:
            raise SecurityError(f"Value too small: {numeric_value} < {min_val}")
        
        if max_val is not None and numeric_value > max_val:
            raise SecurityError(f"Value too large: {numeric_value} > {max_val}")
        
        return numeric_value


class SecureTokenManager:
    """Secure token generation and validation."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize token manager with secret key.
        
        Args:
            secret_key: Secret key for HMAC. If None, generates a new one.
        """
        if secret_key:
            self.secret_key = secret_key.encode('utf-8')
        else:
            self.secret_key = secrets.token_bytes(32)
    
    def generate_token(self, payload: str = None) -> str:
        """
        Generate a secure token.
        
        Args:
            payload: Optional payload to include in token
            
        Returns:
            Secure token string
        """
        if payload is None:
            payload = secrets.token_urlsafe(16)
        
        # Create HMAC signature
        signature = hmac.new(
            self.secret_key,
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Combine payload and signature
        token = f"{payload}.{signature}"
        
        security_logger.info("Secure token generated")
        return token
    
    def verify_token(self, token: str) -> bool:
        """
        Verify a token's authenticity.
        
        Args:
            token: Token to verify
            
        Returns:
            True if token is valid, False otherwise
        """
        try:
            if not token or '.' not in token:
                return False
            
            payload, signature = token.rsplit('.', 1)
            
            # Generate expected signature
            expected_signature = hmac.new(
                self.secret_key,
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Use constant-time comparison to prevent timing attacks
            is_valid = hmac.compare_digest(signature, expected_signature)
            
            if is_valid:
                security_logger.info("Token verified successfully")
            else:
                security_logger.warning("Token verification failed")
            
            return is_valid
            
        except Exception as e:
            security_logger.error(f"Token verification error: {e}")
            return False


class ResourceLimiter:
    """Resource usage limiting and monitoring."""
    
    def __init__(self, max_memory_mb: int = 8000, max_files: int = 10000):
        """
        Initialize resource limiter.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            max_files: Maximum number of files to process
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_files = max_files
        self.files_processed = 0
    
    def check_memory_usage(self) -> bool:
        """
        Check if current memory usage is within limits.
        
        Returns:
            True if within limits, False otherwise
        """
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss
            
            if memory_usage > self.max_memory_bytes:
                security_logger.warning(
                    f"Memory usage exceeded: {memory_usage / 1024 / 1024:.1f}MB "
                    f"> {self.max_memory_bytes / 1024 / 1024}MB"
                )
                return False
                
            return True
        except ImportError:
            # psutil not available, skip check
            return True
        except Exception as e:
            security_logger.error(f"Memory check failed: {e}")
            return True  # Fail safe
    
    def check_file_limit(self) -> bool:
        """
        Check if file processing limit is exceeded.
        
        Returns:
            True if within limits, False otherwise
        """
        if self.files_processed >= self.max_files:
            security_logger.warning(
                f"File limit exceeded: {self.files_processed} >= {self.max_files}"
            )
            return False
        return True
    
    def increment_file_count(self):
        """Increment the processed file counter."""
        self.files_processed += 1


# Global instances for easy access
default_path_validator = PathValidator()
default_input_validator = InputValidator()
default_resource_limiter = ResourceLimiter()


def get_secure_temp_dir() -> str:
    """
    Get a secure temporary directory path.
    
    Returns:
        Path to secure temp directory
    """
    temp_dir = os.path.join(os.getcwd(), "temp", "loopy_comfy_secure")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe filesystem operations.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace unsafe characters
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:250] + ext
    
    # Ensure it's not empty or just dots
    if not sanitized or sanitized.replace('.', '').replace('_', '') == '':
        sanitized = f"file_{secrets.token_hex(4)}.tmp"
    
    return sanitized