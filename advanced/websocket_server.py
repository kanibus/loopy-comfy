# -*- coding: utf-8 -*-
"""
Enhanced WebSocket Server for LoopyComfy v2.0

This module provides WebSocket-based real-time preview with authentication,
browser fallbacks, and adaptive encoding.
"""

import asyncio
import websockets
import json
import time
import base64
import io
import sys
import os
from urllib.parse import urlparse, parse_qs
from typing import Set, Dict, Any, Optional, List
import numpy as np

# Add path for security utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.security_utils import SecureTokenManager, SecurityError


class VideoEncoder:
    """Base class for video encoders with fallback support."""
    
    def __init__(self):
        self.supports_h264 = False
        self.supports_h265 = False
        self.supports_webp = False
        self.supports_jpeg = True  # Always supported
    
    async def encode_jpeg(self, frame: np.ndarray, quality: int = 85) -> str:
        """Encode frame as JPEG (always available fallback)."""
        try:
            from PIL import Image
            
            # Convert to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            image = Image.fromarray(frame, 'RGB')
            
            # Encode to JPEG
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality, optimize=True)
            
            # Encode to base64
            jpeg_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return jpeg_data
            
        except Exception as e:
            print(f"JPEG encoding failed: {e}")
            raise
    
    async def encode_webp(self, frame: np.ndarray, quality: int = 85) -> Optional[str]:
        """Encode frame as WebP if supported."""
        try:
            from PIL import Image
            
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            image = Image.fromarray(frame, 'RGB')
            
            buffer = io.BytesIO()
            image.save(buffer, format='WEBP', quality=quality, optimize=True)
            
            webp_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            self.supports_webp = True
            
            return webp_data
            
        except Exception:
            self.supports_webp = False
            return None
    
    async def encode_h264(self, frame: np.ndarray) -> Optional[bytes]:
        """Encode frame as H.264 if hardware encoder available."""
        try:
            # This would require hardware encoding support
            # For now, return None to use fallback
            return None
            
        except Exception:
            self.supports_h264 = False
            return None


class WebSocketPreviewServer:
    """WebSocket server with authentication and browser fallback."""
    
    def __init__(self, config):
        """Initialize WebSocket server with secure authentication."""
        self.config = config
        self.port = config.get('real_time.websocket_port', 8765)
        self.auth_required = config.get('real_time.websocket_auth', True)
        
        # Security: Initialize secure token manager
        secret_key = config.get('real_time.auth_token', None)
        self.token_manager = SecureTokenManager(secret_key)
        
        # Rate limiting for authentication attempts
        self.auth_attempts = {}  # IP -> [timestamp, ...]
        self.max_auth_attempts = 5
        self.auth_window_seconds = 300  # 5 minutes
        
        # Client management
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.authenticated_clients: Dict[websockets.WebSocketServerProtocol, Dict] = {}
        
        # Connection rate limiting
        self.connection_attempts = {}  # IP -> [timestamp, ...]
        self.max_connections_per_ip = 3
        
        # Encoder
        self.encoder = VideoEncoder()
        
        # Server instance
        self.server = None
        self.is_running = False
        
        # Performance tracking
        self.frames_sent = 0
        self.bytes_sent = 0
        self.start_time = time.time()
        
        print(f"Secure WebSocket server initialized on port {self.port}")
    
    async def start(self):
        """Start WebSocket server."""
        try:
            self.server = await websockets.serve(
                self.handle_client,
                "localhost",
                self.port,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
                max_size=10 * 1024 * 1024  # 10MB max message size
            )
            
            self.is_running = True
            self.start_time = time.time()
            
            print(f"WebSocket server started on ws://localhost:{self.port}")
            
        except Exception as e:
            print(f"Failed to start WebSocket server: {e}")
            raise
    
    async def stop(self):
        """Stop WebSocket server."""
        self.is_running = False
        
        # Close all client connections
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients.copy()],
                return_exceptions=True
            )
        
        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        print("WebSocket server stopped")
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections with authentication."""
        client_info = {
            'connected_at': time.time(),
            'frames_received': 0,
            'last_ping': time.time(),
            'capabilities': {},
            'preferred_encoding': 'jpeg'
        }
        
        try:
            # Parse authentication from URL
            query = parse_qs(urlparse(path).query)
            token = query.get('token', [None])[0]
            
            # Check authentication
            if self.auth_required:
                if not token or not self._verify_token(token):
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Authentication required',
                        'code': 'AUTH_REQUIRED'
                    }))
                    await websocket.close(code=1008, reason="Unauthorized")
                    return
            
            # Add to client lists
            self.clients.add(websocket)
            client_info['token'] = token
            self.authenticated_clients[websocket] = client_info
            
            print(f"Client connected from {websocket.remote_address}")
            
            # Send initial capabilities
            await self._send_capabilities(websocket)
            
            # Handle client messages
            async for message in websocket:
                await self._handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosedOK:
            pass
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Client connection error: {e}")
        except Exception as e:
            print(f"WebSocket handler error: {e}")
        finally:
            # Cleanup
            self.clients.discard(websocket)
            self.authenticated_clients.pop(websocket, None)
            print(f"Client disconnected")
    
    async def _send_capabilities(self, websocket):
        """Send server capabilities to client."""
        capabilities = {
            'type': 'capabilities',
            'server_version': '2.0.0',
            'encodings': {
                'jpeg': True,
                'webp': self.encoder.supports_webp,
                'h264': self.encoder.supports_h264
            },
            'features': {
                'real_time_preview': True,
                'quality_adaptation': True,
                'performance_metrics': True,
                'fallback_encodings': True
            },
            'max_resolution': [1920, 1080],
            'supported_fps': [15, 30, 60]
        }
        
        await websocket.send(json.dumps(capabilities))
    
    async def _handle_client_message(self, websocket, message):
        """Handle messages from client."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'capabilities':
                # Store client capabilities
                client_info = self.authenticated_clients.get(websocket, {})
                client_info['capabilities'] = data.get('capabilities', {})
                
                # Determine best encoding
                if data.get('capabilities', {}).get('supports_webp'):
                    client_info['preferred_encoding'] = 'webp'
                elif data.get('capabilities', {}).get('supports_h264'):
                    client_info['preferred_encoding'] = 'h264'
                else:
                    client_info['preferred_encoding'] = 'jpeg'
                
                await websocket.send(json.dumps({
                    'type': 'capabilities_ack',
                    'preferred_encoding': client_info['preferred_encoding']
                }))
            
            elif message_type == 'ping':
                # Update ping time
                client_info = self.authenticated_clients.get(websocket, {})
                client_info['last_ping'] = time.time()
                
                await websocket.send(json.dumps({
                    'type': 'pong',
                    'timestamp': time.time()
                }))
            
            elif message_type == 'quality_request':
                # Handle quality change request
                requested_quality = data.get('quality', 'medium')
                await self._handle_quality_request(websocket, requested_quality)
            
            elif message_type == 'stats_request':
                # Send performance statistics
                stats = self._get_server_stats()
                await websocket.send(json.dumps({
                    'type': 'stats',
                    'data': stats
                }))
        
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Invalid JSON message'
            }))
        except Exception as e:
            print(f"Message handling error: {e}")
    
    async def _handle_quality_request(self, websocket, quality: str):
        """Handle client quality change request."""
        client_info = self.authenticated_clients.get(websocket, {})
        client_info['requested_quality'] = quality
        
        await websocket.send(json.dumps({
            'type': 'quality_ack',
            'quality': quality,
            'message': f'Quality set to {quality}'
        }))
    
    def _verify_token(self, token: str, client_ip: str = None) -> bool:
        """
        Securely verify authentication token with rate limiting.
        
        Args:
            token: Authentication token to verify
            client_ip: Client IP address for rate limiting
            
        Returns:
            True if token is valid and rate limits not exceeded
        """
        if not token or not isinstance(token, str):
            return False
        
        # Rate limiting for authentication attempts
        if client_ip and self._is_rate_limited(client_ip):
            return False
        
        try:
            # Use secure token verification
            is_valid = self.token_manager.verify_token(token)
            
            # Record authentication attempt
            if client_ip:
                self._record_auth_attempt(client_ip, success=is_valid)
            
            return is_valid
            
        except Exception as e:
            print(f"Token verification error: {e}")
            return False
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client IP is rate limited for authentication attempts."""
        now = time.time()
        
        # Clean old attempts
        if client_ip in self.auth_attempts:
            self.auth_attempts[client_ip] = [
                timestamp for timestamp in self.auth_attempts[client_ip]
                if now - timestamp < self.auth_window_seconds
            ]
        
        # Check current attempt count
        attempt_count = len(self.auth_attempts.get(client_ip, []))
        return attempt_count >= self.max_auth_attempts
    
    def _record_auth_attempt(self, client_ip: str, success: bool) -> None:
        """Record authentication attempt for rate limiting."""
        now = time.time()
        
        if client_ip not in self.auth_attempts:
            self.auth_attempts[client_ip] = []
        
        # Only record failed attempts for rate limiting
        if not success:
            self.auth_attempts[client_ip].append(now)
    
    def _is_connection_rate_limited(self, client_ip: str) -> bool:
        """Check if client IP has too many connections."""
        now = time.time()
        
        # Clean old connection attempts (last 60 seconds)
        if client_ip in self.connection_attempts:
            self.connection_attempts[client_ip] = [
                timestamp for timestamp in self.connection_attempts[client_ip]
                if now - timestamp < 60
            ]
        
        connection_count = len(self.connection_attempts.get(client_ip, []))
        return connection_count >= self.max_connections_per_ip
    
    def _record_connection_attempt(self, client_ip: str) -> None:
        """Record connection attempt for rate limiting."""
        now = time.time()
        
        if client_ip not in self.connection_attempts:
            self.connection_attempts[client_ip] = []
        
        self.connection_attempts[client_ip].append(now)
    
    async def broadcast_frame(self, frame: np.ndarray, metadata: Optional[Dict] = None):
        """Broadcast frame to all connected clients."""
        if not self.clients:
            return
        
        # Prepare encodings
        encoded_data = {}
        
        try:
            # Always prepare JPEG fallback
            encoded_data['jpeg'] = await self.encoder.encode_jpeg(frame, quality=85)
            
            # Prepare WebP if supported
            webp_data = await self.encoder.encode_webp(frame, quality=85)
            if webp_data:
                encoded_data['webp'] = webp_data
            
            # H.264 would go here if supported
            
        except Exception as e:
            print(f"Frame encoding error: {e}")
            return
        
        # Broadcast to clients
        disconnected_clients = set()
        
        for client in self.clients.copy():
            try:
                client_info = self.authenticated_clients.get(client, {})
                preferred_encoding = client_info.get('preferred_encoding', 'jpeg')
                
                # Choose best available encoding
                if preferred_encoding in encoded_data:
                    encoding = preferred_encoding
                    data = encoded_data[preferred_encoding]
                else:
                    encoding = 'jpeg'
                    data = encoded_data['jpeg']
                
                # Prepare message
                message = {
                    'type': 'frame',
                    'encoding': encoding,
                    'data': data,
                    'timestamp': time.time(),
                    'frame_id': self.frames_sent
                }
                
                # Add metadata if provided
                if metadata:
                    message['metadata'] = metadata
                
                # Send to client
                await client.send(json.dumps(message))
                
                # Update client stats
                client_info['frames_received'] = client_info.get('frames_received', 0) + 1
                
                # Track bytes sent (approximate)
                self.bytes_sent += len(data)
                
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                print(f"Failed to send frame to client: {e}")
                disconnected_clients.add(client)
        
        # Clean up disconnected clients
        for client in disconnected_clients:
            self.clients.discard(client)
            self.authenticated_clients.pop(client, None)
        
        # Update frame counter
        self.frames_sent += 1
        
        # Log performance periodically
        if self.frames_sent % 300 == 0:  # Every 10 seconds at 30fps
            self._log_performance()
    
    def _log_performance(self):
        """Log performance metrics."""
        uptime = time.time() - self.start_time
        fps = self.frames_sent / uptime if uptime > 0 else 0
        mbps = (self.bytes_sent * 8 / 1_000_000) / uptime if uptime > 0 else 0
        
        print(f"WebSocket Server Stats:")
        print(f"  Clients: {len(self.clients)}")
        print(f"  Frames sent: {self.frames_sent}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Bandwidth: {mbps:.2f} Mbps")
        print(f"  Uptime: {uptime:.1f}s")
    
    def _get_server_stats(self) -> Dict[str, Any]:
        """Get current server statistics."""
        uptime = time.time() - self.start_time
        fps = self.frames_sent / uptime if uptime > 0 else 0
        mbps = (self.bytes_sent * 8 / 1_000_000) / uptime if uptime > 0 else 0
        
        return {
            'uptime': uptime,
            'connected_clients': len(self.clients),
            'frames_sent': self.frames_sent,
            'bytes_sent': self.bytes_sent,
            'fps': fps,
            'bandwidth_mbps': mbps,
            'encodings_supported': {
                'jpeg': True,
                'webp': self.encoder.supports_webp,
                'h264': self.encoder.supports_h264
            },
            'port': self.port,
            'auth_required': self.auth_required
        }
    
    def get_client_count(self) -> int:
        """Get number of connected clients."""
        return len(self.clients)
    
    def get_client_info(self) -> List[Dict]:
        """Get information about connected clients."""
        client_info = []
        
        for client, info in self.authenticated_clients.items():
            client_info.append({
                'address': str(client.remote_address),
                'connected_at': info['connected_at'],
                'frames_received': info.get('frames_received', 0),
                'preferred_encoding': info.get('preferred_encoding', 'jpeg'),
                'last_ping': info.get('last_ping', 0),
                'capabilities': info.get('capabilities', {})
            })
        
        return client_info
    
    async def send_notification(self, notification_type: str, message: str, data: Optional[Dict] = None):
        """Send notification to all clients."""
        notification = {
            'type': 'notification',
            'notification_type': notification_type,
            'message': message,
            'timestamp': time.time()
        }
        
        if data:
            notification['data'] = data
        
        # Send to all clients
        if self.clients:
            await asyncio.gather(
                *[client.send(json.dumps(notification)) for client in self.clients.copy()],
                return_exceptions=True
            )
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        if self.is_running:
            try:
                asyncio.create_task(self.stop())
            except Exception:
                pass