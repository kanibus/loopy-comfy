"""
VideoSaver Node for Loopy Comfy

This node saves composed video frames to video files using FFmpeg encoding
with various codec and quality options.
"""

import os
import cv2
import numpy as np
import ffmpeg
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


class LoopyComfy_VideoSaver:
    """
    ComfyUI node for saving video frames to video files.
    
    Takes frame arrays from VideoSequenceComposer and encodes them to video files
    using FFmpeg with configurable codec and quality settings.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input types for ComfyUI interface."""
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Frame tensor from VideoSequenceComposer"
                }),
                "output_filename": ("STRING", {
                    "default": "loopy_comfy_output.mp4",
                    "multiline": False,
                    "placeholder": "Output filename (e.g., avatar.mp4)"
                }),
                "output_directory": ("STRING", {
                    "default": "./output/",
                    "multiline": False,
                    "placeholder": "Output directory path"
                }),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Output frame rate"
                }),
                "codec": (["libx264", "libx265", "mpeg4"], {
                    "default": "libx264",
                    "tooltip": "Video codec for encoding"
                }),
                "quality": ("INT", {
                    "default": 23,
                    "min": 0,
                    "max": 51,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "CRF quality (lower = better quality, larger file)"
                })
            },
            "optional": {
                "pixel_format": (["yuv420p", "yuv444p", "rgb24"], {
                    "default": "yuv420p",
                    "tooltip": "Output pixel format"
                }),
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Overwrite existing",
                    "label_off": "Skip if exists",
                    "tooltip": "Overwrite existing output files"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STATISTICS")
    RETURN_NAMES = ("output_path", "save_statistics")
    FUNCTION = "save_video"
    CATEGORY = "video/avatar"
    
    def save_video(
        self,
        frames: np.ndarray,
        output_filename: str,
        output_directory: str,
        fps: float,
        codec: str,
        quality: int,
        pixel_format: str = "yuv420p",
        overwrite: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Save frame array to video file using FFmpeg.
        
        Args:
            frames: Frame array in ComfyUI IMAGE format (B,H,W,C) with values 0-1
            output_filename: Output filename
            output_directory: Output directory path
            fps: Output frame rate
            codec: Video codec
            quality: CRF quality setting
            pixel_format: Output pixel format
            overwrite: Whether to overwrite existing files
            
        Returns:
            Tuple containing (output_path, statistics)
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If encoding fails
        """
        try:
            # Validate inputs
            if frames.size == 0:
                raise ValueError("No frames provided")
            
            if fps <= 0:
                raise ValueError("FPS must be positive")
            
            # Ensure output directory exists
            os.makedirs(output_directory, exist_ok=True)
            
            # Construct full output path
            output_path = os.path.join(output_directory, output_filename)
            output_path = os.path.abspath(output_path)
            
            # Check if file exists and handle overwrite
            if os.path.exists(output_path) and not overwrite:
                raise ValueError(f"Output file exists and overwrite is disabled: {output_path}")
            
            # Validate codec availability
            if not self._check_codec_available(codec):
                print(f"Warning: Codec {codec} not available, falling back to libx264")
                codec = "libx264"
            
            # Convert frames format for FFmpeg
            processed_frames = self._prepare_frames_for_encoding(frames)
            
            # Encode video using FFmpeg
            statistics = self._encode_video_ffmpeg(
                processed_frames, output_path, fps, codec, quality, pixel_format
            )
            
            # Add output information to statistics
            file_stats = os.stat(output_path)
            statistics.update({
                "output_path": output_path,
                "output_filename": output_filename,
                "output_directory": output_directory,
                "file_size": file_stats.st_size,
                "file_size_mb": file_stats.st_size / (1024 * 1024),
                "encoding_parameters": {
                    "codec": codec,
                    "fps": fps,
                    "quality": quality,
                    "pixel_format": pixel_format
                }
            })
            
            print(f"Video saved successfully: {output_path}")
            print(f"File size: {statistics['file_size_mb']:.1f}MB")
            
            return (output_path, statistics)
            
        except Exception as e:
            raise RuntimeError(f"Failed to save video: {str(e)}")
    
    def _prepare_frames_for_encoding(self, frames: np.ndarray) -> np.ndarray:
        """
        Prepare frames for FFmpeg encoding.
        
        Args:
            frames: ComfyUI IMAGE tensor (B,H,W,C) with values 0-1
            
        Returns:
            Frame array ready for encoding (uint8, 0-255)
        """
        # Convert from 0-1 float to 0-255 uint8
        if frames.dtype == np.float32 or frames.dtype == np.float64:
            frames = (frames * 255).astype(np.uint8)
        
        # Ensure uint8 format
        frames = frames.astype(np.uint8)
        
        return frames
    
    def _encode_video_ffmpeg(
        self,
        frames: np.ndarray,
        output_path: str,
        fps: float,
        codec: str,
        quality: int,
        pixel_format: str
    ) -> Dict[str, Any]:
        """
        Encode video frames using FFmpeg.
        
        Args:
            frames: Frame array (B,H,W,C) uint8
            output_path: Output file path
            fps: Frame rate
            codec: Video codec
            quality: CRF quality
            pixel_format: Pixel format
            
        Returns:
            Encoding statistics
        """
        batch_size, height, width, channels = frames.shape
        
        # Create FFmpeg input stream
        input_stream = ffmpeg.input(
            'pipe:',
            format='rawvideo',
            pix_fmt='rgb24',
            s=f'{width}x{height}',
            r=fps
        )
        
        # Configure output stream
        output_args = {
            'c:v': codec,
            'crf': quality,
            'pix_fmt': pixel_format,
            'movflags': '+faststart',  # Enable fast start for web compatibility
        }
        
        # Add codec-specific optimizations
        if codec == 'libx264':
            output_args.update({
                'preset': 'medium',
                'profile:v': 'high',
                'level': '4.0'
            })
        elif codec == 'libx265':
            output_args.update({
                'preset': 'medium',
                'profile:v': 'main'
            })
        
        output_stream = ffmpeg.output(
            input_stream,
            output_path,
            **output_args
        )
        
        # Overwrite output file if it exists
        output_stream = ffmpeg.overwrite_output(output_stream)
        
        try:
            # Run FFmpeg encoding
            process = ffmpeg.run_async(
                output_stream,
                pipe_stdin=True,
                pipe_stderr=True,
                quiet=True
            )
            
            # Write frames to FFmpeg stdin
            for i, frame in enumerate(frames):
                frame_bytes = frame.tobytes()
                process.stdin.write(frame_bytes)
                
                # Progress indication
                if i % 100 == 0:
                    progress = (i + 1) / len(frames) * 100
                    print(f"Encoding progress: {progress:.1f}%")
            
            # Close stdin and wait for completion
            process.stdin.close()
            stderr_output = process.stderr.read().decode('utf-8')
            return_code = process.wait()
            
            if return_code != 0:
                raise RuntimeError(f"FFmpeg encoding failed: {stderr_output}")
            
            # Calculate statistics
            statistics = {
                "frame_count": len(frames),
                "duration_seconds": len(frames) / fps,
                "resolution": f"{width}x{height}",
                "encoding_success": True,
                "ffmpeg_stderr": stderr_output
            }
            
            return statistics
            
        except Exception as e:
            if 'process' in locals():
                process.terminate()
            raise RuntimeError(f"FFmpeg encoding error: {str(e)}")
    
    def _check_codec_available(self, codec: str) -> bool:
        """
        Check if codec is available in the system.
        
        Args:
            codec: Codec name
            
        Returns:
            True if available, False otherwise
        """
        try:
            # Use a valid test pattern for codec validation
            # Generate 1 frame of test video data using color pattern
            result = ffmpeg.run(
                ffmpeg.input('color=c=black:size=320x240:duration=0.1', f='lavfi')
                .output('pipe:', format='null', vcodec=codec, loglevel='quiet'),
                capture_stdout=True,
                capture_stderr=True
            )
            return True
        except ffmpeg.Error as e:
            # If it's specifically a codec error, return False
            if b'Unknown encoder' in e.stderr or b'does not support' in e.stderr:
                return False
            # For other FFmpeg errors, assume codec is available but other issue occurred
            return True
        except Exception:
            # For non-FFmpeg errors, be conservative and return False
            return False
    
    def _validate_disk_space(self, output_path: str, estimated_size: int) -> bool:
        """
        Validate that sufficient disk space is available.
        
        Args:
            output_path: Output file path
            estimated_size: Estimated file size in bytes
            
        Returns:
            True if sufficient space available
        """
        try:
            stat = os.statvfs(os.path.dirname(output_path))
            available_space = stat.f_bavail * stat.f_frsize
            
            # Require 20% buffer above estimated size
            required_space = estimated_size * 1.2
            
            return available_space >= required_space
        except:
            # If can't check, assume it's okay
            return True


# Required for ComfyUI node registration
WEB_DIRECTORY = "./web"