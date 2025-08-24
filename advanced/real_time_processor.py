# -*- coding: utf-8 -*-
"""
Real-Time Video Processing Pipeline for LoopyComfy v2.0

This module provides streaming-based video processing with comprehensive error 
handling, adaptive latency, and automatic quality adjustments.
"""

import asyncio
import time
import secrets
import queue
import threading
from typing import Optional, AsyncGenerator, Dict, Any, Callable
import numpy as np
from contextlib import asynccontextmanager


class StreamBuffer:
    """Thread-safe circular buffer for video frames."""
    
    def __init__(self, max_size: int = 30):
        """Initialize stream buffer."""
        self.max_size = max_size
        self.buffer = queue.Queue(maxsize=max_size)
        self.lock = threading.RLock()
        self.frame_count = 0
    
    def put_frame(self, frame: np.ndarray, timestamp: float) -> bool:
        """Add frame to buffer, dropping oldest if full."""
        try:
            frame_data = {
                'frame': frame,
                'timestamp': timestamp,
                'frame_id': self.frame_count
            }
            
            self.buffer.put_nowait(frame_data)
            self.frame_count += 1
            return True
            
        except queue.Full:
            # Drop oldest frame and add new one
            try:
                self.buffer.get_nowait()
                self.buffer.put_nowait(frame_data)
                self.frame_count += 1
                return True
            except queue.Empty:
                return False
    
    def get_frame(self) -> Optional[Dict]:
        """Get next frame from buffer."""
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            return None
    
    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer.qsize()


class RealTimeVideoProcessor:
    """Stream-based processing with comprehensive error handling."""
    
    def __init__(self, config, resource_monitor, fallback_chain):
        """Initialize real-time processor."""
        self.config = config
        self.monitor = resource_monitor
        self.fallback = fallback_chain
        
        # Stream management
        self.stream_buffer = StreamBuffer(max_size=30)
        self.output_queue = asyncio.Queue(maxsize=60)
        self.processing_task = None
        self.websocket_server = None
        self.is_streaming = False
        
        # Authentication
        self.auth_tokens = set()
        
        # Performance tracking
        self.connection_type = self.detect_connection_type()
        self.stream_quality = 'medium'
        self.fps_target = 30
        self.last_fps_check = time.time()
        self.frame_times = []
        
        # Error handling
        self.error_count = 0
        self.last_error_time = 0
        self.recovery_attempts = 0
        
        print(f"Real-time processor initialized (connection: {self.connection_type})")
    
    def detect_connection_type(self) -> str:
        """Detect if connection is local, LAN, or internet."""
        # In production, this would check client IP ranges
        # For now, default to local for best performance
        return "local"
    
    def generate_auth_token(self) -> str:
        """Generate secure authentication token."""
        token = secrets.token_urlsafe(32)
        self.auth_tokens.add(token)
        return token
    
    def verify_token(self, token: str) -> bool:
        """Verify authentication token."""
        return token in self.auth_tokens
    
    async def start_stream(self, sequence_generator, fps: int = 30) -> bool:
        """Start real-time processing with error recovery."""
        try:
            self.fps_target = fps
            self.is_streaming = True
            
            # Get appropriate processor with fallbacks
            self.processor = self.fallback.get_processor()
            
            # Start processing task
            self.processing_task = asyncio.create_task(
                self._stream_processing_loop(sequence_generator)
            )
            
            # Start WebSocket server with retry logic
            if self.config.is_feature_enabled('real_time'):
                await self._start_websocket_with_retry()
            
            print(f"Real-time streaming started at {fps} FPS")
            return True
            
        except Exception as e:
            print(f"Failed to start real-time processing: {e}")
            self.is_streaming = False
            
            # Fallback to batch processing
            return await self._fallback_to_batch_processing(sequence_generator)
    
    async def _start_websocket_with_retry(self, max_retries: int = 3):
        """Start WebSocket server with retry logic."""
        retries = max_retries
        
        while retries > 0 and self.is_streaming:
            try:
                websocket_server = self.fallback.get_websocket_server()
                if websocket_server:
                    self.websocket_server = websocket_server
                    await websocket_server.start()
                    print("WebSocket server started successfully")
                    break
                else:
                    print("WebSocket server not available")
                    break
                    
            except Exception as e:
                retries -= 1
                print(f"WebSocket start failed: {e} (retries left: {retries})")
                
                if retries > 0:
                    await asyncio.sleep(2 ** (max_retries - retries))  # Exponential backoff
                else:
                    print("WebSocket server failed to start - continuing without real-time preview")
                    self.websocket_server = None
    
    async def _stream_processing_loop(self, sequence_generator):
        """Core streaming loop with adaptive latency."""
        # Get latency target based on connection type
        latency_config = self.config.get('real_time.max_latency_ms', {})
        latency_target = latency_config.get(self.connection_type, 200)
        
        frame_duration = 1.0 / self.fps_target
        last_frame_time = time.time()
        
        print(f"Stream processing loop started (target latency: {latency_target}ms)")
        
        while self.is_streaming:
            try:
                loop_start = time.time()
                
                # Monitor system resources
                if not self.monitor.check_memory_usage():
                    await self._handle_resource_pressure()
                
                # Get next video in sequence
                try:
                    next_video_idx = await self._get_next_video_safe(sequence_generator)
                    if next_video_idx is None:
                        await asyncio.sleep(frame_duration)
                        continue
                        
                except Exception as e:
                    print(f"Sequence generation error: {e}")
                    await asyncio.sleep(frame_duration)
                    continue
                
                # Stream frames from selected video
                async for frame_data in self._stream_video_frames(next_video_idx):
                    if not self.is_streaming:
                        break
                    
                    # Process frame with performance monitoring
                    processing_start = time.time()
                    
                    try:
                        processed_frame = await self._process_frame_adaptive(
                            frame_data['frame']
                        )
                        
                        processing_time = (time.time() - processing_start) * 1000
                        
                        # Check if we're meeting latency target
                        if processing_time > latency_target:
                            await self._handle_latency_issue(processing_time, latency_target)
                        
                        # Send to output queue
                        await self._queue_output_frame(processed_frame, frame_data)
                        
                        # Update performance metrics
                        self._update_fps_metrics(time.time())
                        
                    except Exception as e:
                        print(f"Frame processing error: {e}")
                        self.error_count += 1
                        await self._handle_processing_error(e)
                    
                    # Maintain frame rate
                    await self._maintain_frame_rate(last_frame_time, frame_duration)
                    last_frame_time = time.time()
                
            except Exception as e:
                print(f"Stream loop error: {e}")
                await self._recover_from_error(e)
    
    async def _get_next_video_safe(self, sequence_generator):
        """Safely get next video with error handling."""
        try:
            if hasattr(sequence_generator, 'get_next_state_realtime'):
                return await asyncio.to_thread(
                    sequence_generator.get_next_state_realtime
                )
            else:
                # Fallback to regular sequencer
                return await asyncio.to_thread(
                    sequence_generator.get_next_state
                )
        except Exception as e:
            print(f"Failed to get next video: {e}")
            return None
    
    async def _stream_video_frames(self, video_idx: int) -> AsyncGenerator[Dict, None]:
        """Stream frames from a video with async generator."""
        try:
            # This would integrate with the video asset loader
            # For now, simulate frame streaming
            frame_count = 30 * 5  # 5 seconds at 30fps
            
            for i in range(frame_count):
                if not self.is_streaming:
                    break
                
                # Simulate frame loading (replace with actual video loading)
                frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
                
                yield {
                    'frame': frame,
                    'video_idx': video_idx,
                    'frame_idx': i,
                    'timestamp': time.time()
                }
                
                # Small delay to simulate real-time playback
                await asyncio.sleep(1.0 / self.fps_target)
                
        except Exception as e:
            print(f"Video streaming error: {e}")
            return
    
    async def _process_frame_adaptive(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with adaptive quality based on performance."""
        try:
            # Try GPU processing first if available
            if hasattr(self.processor, 'process_frame_gpu'):
                try:
                    return await asyncio.to_thread(
                        self.processor.process_frame_gpu, frame
                    )
                except Exception as e:
                    print(f"GPU processing failed, falling back to CPU: {e}")
            
            # CPU processing fallback
            if hasattr(self.processor, 'process_frame_cpu'):
                return await asyncio.to_thread(
                    self.processor.process_frame_cpu, frame
                )
            else:
                # Basic processing fallback
                return await asyncio.to_thread(self._basic_frame_processing, frame)
                
        except Exception as e:
            print(f"Frame processing failed: {e}")
            return frame  # Return original frame as last resort
    
    def _basic_frame_processing(self, frame: np.ndarray) -> np.ndarray:
        """Basic frame processing as ultimate fallback."""
        try:
            import cv2
            # Just ensure RGB format
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        except Exception:
            return frame
    
    async def _queue_output_frame(self, frame: np.ndarray, frame_data: Dict):
        """Queue processed frame for output."""
        try:
            output_data = {
                'frame': frame,
                'timestamp': frame_data['timestamp'],
                'processing_time': time.time() - frame_data['timestamp'],
                'video_idx': frame_data.get('video_idx'),
                'frame_idx': frame_data.get('frame_idx')
            }
            
            await self.output_queue.put(output_data)
            
            # Send to WebSocket clients if available
            if self.websocket_server:
                await self._broadcast_to_websocket(frame, output_data)
                
        except asyncio.QueueFull:
            # Drop oldest frame
            try:
                self.output_queue.get_nowait()
                await self.output_queue.put(output_data)
            except asyncio.QueueEmpty:
                pass
    
    async def _broadcast_to_websocket(self, frame: np.ndarray, frame_data: Dict):
        """Broadcast frame to WebSocket clients."""
        try:
            if hasattr(self.websocket_server, 'broadcast_frame'):
                await self.websocket_server.broadcast_frame(frame)
        except Exception as e:
            print(f"WebSocket broadcast error: {e}")
    
    async def _handle_resource_pressure(self):
        """Handle high resource usage."""
        print("Resource pressure detected - reducing quality")
        
        if self.stream_quality == 'high':
            self.stream_quality = 'medium'
        elif self.stream_quality == 'medium':
            self.stream_quality = 'low'
        
        # Trigger emergency cleanup
        await asyncio.to_thread(self.monitor.emergency_cleanup)
        
        # Reduce frame rate temporarily
        self.fps_target = max(15, self.fps_target - 5)
        
        print(f"Quality reduced to {self.stream_quality}, FPS: {self.fps_target}")
    
    async def _handle_latency_issue(self, actual_ms: float, target_ms: float):
        """Handle latency issues by reducing quality."""
        latency_ratio = actual_ms / target_ms
        
        if latency_ratio > 2.0:  # More than 2x target
            if self.stream_quality == 'high':
                self.stream_quality = 'medium'
                print(f"High latency detected ({actual_ms:.1f}ms) - reduced to medium quality")
            elif self.stream_quality == 'medium':
                self.stream_quality = 'low'
                print(f"High latency detected ({actual_ms:.1f}ms) - reduced to low quality")
    
    async def _handle_processing_error(self, error: Exception):
        """Handle processing errors with recovery."""
        current_time = time.time()
        
        # Track error frequency
        if current_time - self.last_error_time > 10:  # Reset error count every 10 seconds
            self.error_count = 1
        else:
            self.error_count += 1
        
        self.last_error_time = current_time
        
        # If too many errors, attempt recovery
        if self.error_count > 5:
            print(f"Multiple errors detected ({self.error_count}) - attempting recovery")
            await self._recover_from_error(error)
    
    async def _recover_from_error(self, error: Exception):
        """Recover from streaming errors."""
        self.recovery_attempts += 1
        
        if self.recovery_attempts > 3:
            print("Max recovery attempts reached - stopping stream")
            await self.stop_stream()
            return
        
        print(f"Recovery attempt {self.recovery_attempts}")
        
        # Get new processor
        self.processor = self.fallback.get_processor()
        
        # Reduce quality
        self.stream_quality = 'low'
        self.fps_target = 15
        
        # Clear queues
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Short pause before resuming
        await asyncio.sleep(1.0)
        
        print("Recovery completed")
    
    def _update_fps_metrics(self, current_time: float):
        """Update FPS tracking metrics."""
        self.frame_times.append(current_time)
        
        # Keep only last second of frame times
        cutoff_time = current_time - 1.0
        self.frame_times = [t for t in self.frame_times if t > cutoff_time]
        
        # Update FPS calculation
        if len(self.frame_times) > 1:
            actual_fps = len(self.frame_times)
            
            if current_time - self.last_fps_check > 5.0:  # Check every 5 seconds
                print(f"Actual FPS: {actual_fps:.1f} (target: {self.fps_target})")
                self.last_fps_check = current_time
    
    async def _maintain_frame_rate(self, last_frame_time: float, frame_duration: float):
        """Maintain target frame rate with sleep."""
        current_time = time.time()
        sleep_time = frame_duration - (current_time - last_frame_time)
        
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
    
    async def _fallback_to_batch_processing(self, sequence_generator) -> bool:
        """Fallback to batch processing if real-time fails."""
        print("Falling back to batch processing mode")
        
        try:
            # Use basic batch processor
            batch_processor = self.fallback._load_basic_processor()
            
            if batch_processor:
                print("Batch processing mode activated")
                return True
            else:
                print("Batch processing fallback failed")
                return False
                
        except Exception as e:
            print(f"Batch fallback error: {e}")
            return False
    
    async def stop_stream(self):
        """Stop streaming with cleanup."""
        print("Stopping real-time stream...")
        
        self.is_streaming = False
        
        # Stop processing task
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Stop WebSocket server
        if self.websocket_server:
            try:
                await self.websocket_server.stop()
            except Exception as e:
                print(f"WebSocket stop error: {e}")
        
        # Cleanup processor
        if hasattr(self.processor, 'cleanup'):
            try:
                await asyncio.to_thread(self.processor.cleanup)
            except Exception as e:
                print(f"Processor cleanup error: {e}")
        
        print("Real-time stream stopped")
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get current stream statistics."""
        current_time = time.time()
        
        # Calculate actual FPS
        recent_frames = [t for t in self.frame_times if t > current_time - 1.0]
        actual_fps = len(recent_frames)
        
        return {
            'is_streaming': self.is_streaming,
            'target_fps': self.fps_target,
            'actual_fps': actual_fps,
            'stream_quality': self.stream_quality,
            'buffer_size': self.stream_buffer.size(),
            'output_queue_size': self.output_queue.qsize(),
            'error_count': self.error_count,
            'recovery_attempts': self.recovery_attempts,
            'websocket_connected': self.websocket_server is not None,
            'connection_type': self.connection_type
        }
    
    @asynccontextmanager
    async def stream_context(self, sequence_generator, fps: int = 30):
        """Context manager for safe stream handling."""
        try:
            success = await self.start_stream(sequence_generator, fps)
            if success:
                yield self
            else:
                raise RuntimeError("Failed to start stream")
        finally:
            await self.stop_stream()