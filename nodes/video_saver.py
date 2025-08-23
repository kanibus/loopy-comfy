# -*- coding: utf-8 -*-
"""
VideoSaver Node for Loopy Comfy

This node saves composed video frames to video files using FFmpeg encoding
with various codec and quality options.
"""

import os
import sys
import cv2
import numpy as np
import ffmpeg
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# CRITICAL: Standardized import path setup for ComfyUI compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class LoopyComfy_VideoSaver:
    """
    ComfyUI node for saving video frames to video files.
    
    Takes frame arrays from VideoSequenceComposer and encodes them to video files
    using FFmpeg with configurable codec and quality settings.
    
    Enhanced with platform-specific presets and multi-format export capabilities.
    """
    
    # Platform-specific encoding presets optimized for each platform
    PLATFORM_PRESETS = {
        "YouTube": {"codec": "libx264", "quality": 18, "bitrate": "8M", "preset": "medium", "profile": "high"},
        "Vimeo": {"codec": "libx264", "quality": 17, "bitrate": "10M", "preset": "medium", "profile": "high"},
        "TikTok": {"codec": "libx264", "quality": 20, "bitrate": "6M", "preset": "fast", "profile": "main"},
        "Instagram Reels": {"codec": "libx264", "quality": 21, "bitrate": "5M", "preset": "fast", "profile": "main"},
        "Instagram Feed": {"codec": "libx264", "quality": 22, "bitrate": "4M", "preset": "fast", "profile": "main"},
        "Twitter": {"codec": "libx264", "quality": 22, "bitrate": "5M", "preset": "fast", "profile": "main"},
        "Snapchat": {"codec": "libx264", "quality": 21, "bitrate": "6M", "preset": "fast", "profile": "main"},
        "LinkedIn": {"codec": "libx264", "quality": 20, "bitrate": "7M", "preset": "medium", "profile": "high"},
        "Professional": {"codec": "prores_ks", "bitrate": "200M", "preset": "medium", "profile": "hq", "pix_fmt": "yuv422p10le"},
        "Archive (H.265)": {"codec": "libx265", "quality": 15, "bitrate": "15M", "preset": "medium", "profile": "main"},
        "Web Optimized": {"codec": "libx264", "quality": 23, "bitrate": "6M", "preset": "fast", "profile": "baseline"},
        "High Quality": {"codec": "libx264", "quality": 16, "bitrate": "20M", "preset": "slow", "profile": "high"},
        "Fast Encode": {"codec": "libx264", "quality": 25, "bitrate": "4M", "preset": "ultrafast", "profile": "baseline"},
        "Custom": {}
    }
    
    # Quality presets for quick selection
    QUALITY_PRESETS = {
        "Draft": {"quality": 28, "preset": "ultrafast"},
        "Good": {"quality": 23, "preset": "medium"},
        "Best": {"quality": 18, "preset": "slow"},
        "Lossless": {"quality": 0, "preset": "medium"}
    }
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input types for ComfyUI interface."""
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Frame tensor from VideoSequenceComposer"
                }),
                "platform_preset": (list(cls.PLATFORM_PRESETS.keys()), {
                    "default": "YouTube",
                    "tooltip": "Platform-optimized encoding presets with ideal settings for each service"
                }),
                "output_filename": ("STRING", {
                    "default": "loopy_comfy_output.mp4",
                    "multiline": False,
                    "placeholder": "Use %timestamp% for auto timestamp, %preset% for platform name"
                }),
                "output_directory": ("STRING", {
                    "default": "./output/",
                    "multiline": False,
                    "placeholder": "Output directory path"
                }),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 240.0,
                    "step": 1.0,
                    "display": "slider",
                    "tooltip": "Output frame rate (should match VideoSequenceComposer output)"
                })
            },
            "optional": {
                "quality_override": (["auto", "Draft", "Good", "Best", "Lossless"], {
                    "default": "auto",
                    "tooltip": "Override platform preset quality (auto uses platform optimal)"
                }),
                "multi_format_export": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Export Multiple Formats",
                    "label_off": "Single Format Only",
                    "tooltip": "Export in multiple formats simultaneously"
                }),
                "export_formats": ("STRING", {
                    "default": "mp4,mov,webm",
                    "multiline": False,
                    "placeholder": "Comma-separated formats (mp4,mov,webm,avi)",
                    "tooltip": "Additional formats to export (only when multi_format_export is enabled)"
                }),
                "embed_metadata": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Include Metadata",
                    "label_off": "No Metadata",
                    "tooltip": "Embed title, artist, and creation info in video file"
                }),
                "metadata_title": ("STRING", {
                    "default": "",
                    "placeholder": "Video title for metadata"
                }),
                "metadata_artist": ("STRING", {
                    "default": "",
                    "placeholder": "Creator/artist name"
                }),
                "metadata_comment": ("STRING", {
                    "default": "Generated with LoopyComfy - Non-repetitive video avatars",
                    "multiline": True,
                    "placeholder": "Additional comments or description"
                }),
                "estimate_file_size": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Show Size Estimate",
                    "label_off": "No Estimate",
                    "tooltip": "Calculate and display estimated file size before encoding"
                }),
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Overwrite existing",
                    "label_off": "Skip if exists",
                    "tooltip": "Overwrite existing output files"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STATISTICS", "INT", "FLOAT")
    RETURN_NAMES = ("output_path", "save_statistics", "files_created", "total_size_mb")
    FUNCTION = "save_video"
    CATEGORY = "video/avatar"
    
    def save_video(
        self,
        frames: np.ndarray,
        platform_preset: str,
        output_filename: str,
        output_directory: str,
        fps: float,
        quality_override: str = "auto",
        multi_format_export: bool = False,
        export_formats: str = "mp4,mov,webm",
        embed_metadata: bool = True,
        metadata_title: str = "",
        metadata_artist: str = "",
        metadata_comment: str = "Generated with LoopyComfy",
        estimate_file_size: bool = True,
        overwrite: bool = True,
        **kwargs
    ) -> Tuple[str, Dict[str, Any], int, float]:
        """
        Save frame array to video file using FFmpeg with platform presets.
        
        Args:
            frames: Frame array in ComfyUI IMAGE format (B,H,W,C) with values 0-1
            platform_preset: Selected platform preset for optimized encoding
            output_filename: Output filename with template support
            output_directory: Output directory path
            fps: Output frame rate
            quality_override: Override platform quality setting
            multi_format_export: Export multiple formats
            export_formats: Comma-separated list of formats for multi-export
            embed_metadata: Include metadata in video file
            metadata_*: Metadata fields
            estimate_file_size: Show size estimate before encoding
            overwrite: Whether to overwrite existing files
            
        Returns:
            Tuple containing (primary_output_path, statistics, files_created, total_size_mb)
            
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
            
            # Get platform-specific encoding settings
            encoding_settings = self._get_encoding_settings(platform_preset, quality_override)
            
            # Process filename template
            processed_filename = self._process_filename_template(output_filename, platform_preset)
            
            # Construct primary output path
            primary_output_path = os.path.join(output_directory, processed_filename)
            primary_output_path = os.path.abspath(primary_output_path)
            
            # Check if file exists and handle overwrite
            if os.path.exists(primary_output_path) and not overwrite:
                raise ValueError(f"Output file exists and overwrite is disabled: {primary_output_path}")
            
            # Validate codec availability and security
            codec = self._validate_codec_name(encoding_settings['codec'])
            encoding_settings['codec'] = codec
            
            if not self._check_codec_available(codec):
                print(f"Warning: Codec {codec} not available, falling back to libx264")
                codec = "libx264"
                encoding_settings['codec'] = codec
            
            # Convert frames format for FFmpeg
            processed_frames = self._prepare_frames_for_encoding(frames)
            
            # Estimate file size if requested
            if estimate_file_size:
                estimated_size_mb = self._estimate_file_size(processed_frames, encoding_settings, fps)
                print(f"Estimated file size: {estimated_size_mb:.1f}MB")
            
            # Prepare metadata
            metadata = self._prepare_metadata(embed_metadata, metadata_title, metadata_artist, metadata_comment)
            
            # Encode primary format
            primary_stats = self._encode_video_ffmpeg_enhanced(
                processed_frames, primary_output_path, fps, encoding_settings, metadata
            )
            
            output_files = [primary_output_path]
            total_size_mb = primary_stats['file_size_mb']
            
            # Handle multi-format export
            if multi_format_export and export_formats.strip():
                additional_files, additional_size = self._export_additional_formats(
                    processed_frames, output_directory, processed_filename, fps, 
                    export_formats, encoding_settings, metadata
                )
                output_files.extend(additional_files)
                total_size_mb += additional_size
            
            # Compile comprehensive statistics
            combined_statistics = {
                "primary_output_path": primary_output_path,
                "all_output_files": output_files,
                "files_created": len(output_files),
                "total_size_mb": total_size_mb,
                "platform_preset": platform_preset,
                "encoding_settings": encoding_settings,
                "metadata_embedded": embed_metadata,
                **primary_stats
            }
            
            print(f"Video(s) saved successfully: {len(output_files)} file(s)")
            print(f"Total size: {total_size_mb:.1f}MB")
            
            return (primary_output_path, combined_statistics, len(output_files), total_size_mb)
            
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
    
    # REMOVED: Conflicting _encode_video_ffmpeg method
    # This method was replaced by _encode_video_ffmpeg_enhanced which provides
    # better platform-specific encoding and metadata support
    
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
                capture_stderr=True,
                timeout=30  # 30 seconds timeout for codec validation
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
            import shutil
            available_space = shutil.disk_usage(os.path.dirname(output_path)).free
            
            # Require 20% buffer above estimated size
            required_space = estimated_size * 1.2
            
            return available_space >= required_space
        except:
            # If can't check, assume it's okay
            return True
    
    def _get_encoding_settings(self, platform_preset: str, quality_override: str) -> Dict[str, Any]:
        """
        Get encoding settings from platform preset with optional quality override.
        
        Args:
            platform_preset: Selected platform preset
            quality_override: Quality override setting
            
        Returns:
            Dictionary of encoding settings
        """
        # Start with platform preset
        if platform_preset in self.PLATFORM_PRESETS:
            settings = self.PLATFORM_PRESETS[platform_preset].copy()
        else:
            # Fallback to YouTube settings
            settings = self.PLATFORM_PRESETS["YouTube"].copy()
        
        # Apply quality override if specified
        if quality_override != "auto" and quality_override in self.QUALITY_PRESETS:
            quality_settings = self.QUALITY_PRESETS[quality_override]
            settings.update(quality_settings)
        
        # Ensure required fields have defaults
        settings.setdefault('codec', 'libx264')
        settings.setdefault('quality', 23)
        settings.setdefault('preset', 'medium')
        settings.setdefault('profile', 'high')
        settings.setdefault('pixel_format', 'yuv420p')
        
        return settings
    
    def _process_filename_template(self, filename_template: str, platform_preset: str) -> str:
        """
        Process filename template with variable substitution.
        
        Args:
            filename_template: Filename with template variables
            platform_preset: Selected platform preset
            
        Returns:
            Processed filename
        """
        import time
        
        processed = filename_template
        
        # Replace timestamp
        if '%timestamp%' in processed:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            processed = processed.replace('%timestamp%', timestamp)
        
        # Replace preset name
        if '%preset%' in processed:
            preset_name = platform_preset.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
            processed = processed.replace('%preset%', preset_name)
        
        return processed
    
    def _prepare_metadata(self, embed_metadata: bool, title: str, artist: str, comment: str) -> Dict[str, str]:
        """
        Prepare metadata dictionary for video file.
        
        Args:
            embed_metadata: Whether to include metadata
            title: Video title
            artist: Creator/artist name
            comment: Additional comments
            
        Returns:
            Metadata dictionary
        """
        if not embed_metadata:
            return {}
        
        metadata = {}
        
        if title.strip():
            metadata['title'] = title.strip()
        
        if artist.strip():
            metadata['artist'] = artist.strip()
            metadata['author'] = artist.strip()  # Alternative field
        
        if comment.strip():
            metadata['comment'] = comment.strip()
            metadata['description'] = comment.strip()  # Alternative field
        
        # Add creation timestamp
        import time
        metadata['creation_time'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
        metadata['encoder'] = 'LoopyComfy Non-Linear Video Avatar System'
        
        return metadata
    
    def _estimate_file_size(self, frames: np.ndarray, encoding_settings: Dict[str, Any], fps: float) -> float:
        """
        Estimate output file size based on frames and encoding settings.
        
        Args:
            frames: Frame array
            encoding_settings: Encoding parameters
            fps: Frame rate
            
        Returns:
            Estimated file size in MB
        """
        try:
            frame_count, height, width, channels = frames.shape
            duration_seconds = frame_count / fps
            
            # Get target bitrate (default estimation if not specified)
            bitrate_str = encoding_settings.get('bitrate', '8M')
            
            # Parse bitrate string (e.g., '8M' -> 8000000)
            if bitrate_str.endswith('M'):
                bitrate_bps = float(bitrate_str[:-1]) * 1000000
            elif bitrate_str.endswith('K'):
                bitrate_bps = float(bitrate_str[:-1]) * 1000
            else:
                bitrate_bps = float(bitrate_str)
            
            # Estimate file size: (bitrate * duration) / 8 bits per byte
            estimated_bytes = (bitrate_bps * duration_seconds) / 8
            estimated_mb = estimated_bytes / (1024 * 1024)
            
            return estimated_mb
            
        except Exception as e:
            print(f"Warning: Could not estimate file size: {str(e)}")
            # Rough fallback estimation: ~1MB per 10 seconds for typical settings
            frame_count = frames.shape[0] if len(frames.shape) > 0 else 0
            duration_seconds = frame_count / fps if fps > 0 else 60
            return duration_seconds * 0.1  # Very rough estimate
    
    def _encode_video_ffmpeg_enhanced(
        self,
        frames: np.ndarray,
        output_path: str,
        fps: float,
        encoding_settings: Dict[str, Any],
        metadata: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Enhanced video encoding with platform-specific optimizations and metadata.
        
        Args:
            frames: Frame array (B,H,W,C) uint8
            output_path: Output file path
            fps: Frame rate
            encoding_settings: Platform-specific encoding parameters
            metadata: Metadata to embed
            
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
        
        # Build output arguments from encoding settings
        output_args = {
            'c:v': encoding_settings['codec'],
            'pix_fmt': encoding_settings.get('pixel_format', 'yuv420p'),
            'movflags': '+faststart',  # Enable fast start for web compatibility
        }
        
        # Handle quality settings differently based on codec
        codec = encoding_settings['codec']
        if codec.startswith('prores'):
            # ProRes uses bitrate, not CRF
            if 'bitrate' in encoding_settings:
                output_args['b:v'] = encoding_settings['bitrate']
        else:
            # Most codecs use CRF
            output_args['crf'] = encoding_settings.get('quality', 23)
        
        # Add codec-specific settings
        if 'preset' in encoding_settings:
            output_args['preset'] = encoding_settings['preset']
        
        if 'profile' in encoding_settings and encoding_settings['profile'] != 'auto':
            output_args['profile:v'] = encoding_settings['profile']
        
        if 'bitrate' in encoding_settings:
            output_args['maxrate'] = encoding_settings['bitrate']
            output_args['bufsize'] = encoding_settings['bitrate']
        
        # Add metadata with security sanitization
        for key, value in metadata.items():
            # Sanitize metadata to prevent command injection
            safe_key = self._sanitize_metadata_key(key)
            safe_value = self._sanitize_metadata_value(str(value))
            output_args[f'metadata:{safe_key}'] = safe_value
        
        # Special optimizations per codec
        if codec == 'libx264':
            output_args.setdefault('level', '4.0')
            if 'profile:v' not in output_args:
                output_args['profile:v'] = 'high'
        elif codec == 'libx265':
            output_args.setdefault('tag:v', 'hvc1')  # Better compatibility
        elif codec.startswith('prores'):
            # ProRes specific settings - no CRF, uses bitrate
            if 'crf' in output_args:
                del output_args['crf']  # Remove CRF for ProRes
            output_args.setdefault('vendor', 'ap10')  # Apple ProRes vendor
        
        output_stream = ffmpeg.output(
            input_stream,
            output_path,
            **output_args
        )
        
        # Overwrite output file if it exists
        output_stream = ffmpeg.overwrite_output(output_stream)
        
        try:
            # Run FFmpeg encoding with progress tracking and timeout
            process = ffmpeg.run_async(
                output_stream,
                pipe_stdin=True,
                pipe_stderr=True,
                quiet=True
            )
            
            # Write frames to FFmpeg stdin with progress indication
            for i, frame in enumerate(frames):
                frame_bytes = frame.tobytes()
                process.stdin.write(frame_bytes)
                
                # Progress indication every 5% or 100 frames
                progress_interval = max(10, len(frames) // 20)
                if i % progress_interval == 0 or i == len(frames) - 1:
                    progress = (i + 1) / len(frames) * 100
                    print(f"Encoding progress: {progress:.1f}% ({i+1}/{len(frames)} frames)")
            
            # Close stdin and wait for completion with timeout
            process.stdin.close()
            
            # Add timeout handling for FFmpeg process completion
            import threading
            import time
            
            def read_stderr():
                return process.stderr.read().decode('utf-8')
            
            stderr_thread = threading.Thread(target=read_stderr)
            stderr_thread.daemon = True
            stderr_thread.start()
            
            # Wait for process with timeout (5 minutes for encoding)
            timeout_seconds = 300
            start_time = time.time()
            
            while process.poll() is None:
                if time.time() - start_time > timeout_seconds:
                    process.kill()
                    raise RuntimeError(f"FFmpeg encoding timeout ({timeout_seconds}s) - process killed")
                time.sleep(0.1)
            
            # Get return code and stderr
            return_code = process.returncode
            stderr_thread.join(timeout=1)  # Brief wait for stderr thread
            stderr_output = ""
            try:
                if process.stderr:
                    stderr_output = process.stderr.read().decode('utf-8')
            except:
                stderr_output = "Unable to read stderr"
            
            if return_code != 0:
                raise RuntimeError(f"FFmpeg encoding failed (code {return_code}): {stderr_output}")
            
            # Get file statistics
            file_stats = os.stat(output_path)
            
            # Calculate comprehensive statistics
            statistics = {
                "frame_count": len(frames),
                "duration_seconds": len(frames) / fps,
                "resolution": f"{width}x{height}",
                "file_size": file_stats.st_size,
                "file_size_mb": file_stats.st_size / (1024 * 1024),
                "encoding_success": True,
                "codec_used": codec,
                "quality_setting": encoding_settings['quality'],
                "preset_used": encoding_settings.get('preset', 'unknown'),
                "ffmpeg_stderr": stderr_output[-500:] if len(stderr_output) > 500 else stderr_output  # Last 500 chars
            }
            
            return statistics
            
        except Exception as e:
            if 'process' in locals():
                try:
                    process.terminate()
                except:
                    pass
            raise RuntimeError(f"FFmpeg encoding error: {str(e)}")
    
    def _export_additional_formats(
        self,
        frames: np.ndarray,
        output_directory: str,
        base_filename: str,
        fps: float,
        export_formats: str,
        base_encoding_settings: Dict[str, Any],
        metadata: Dict[str, str]
    ) -> Tuple[List[str], float]:
        """
        Export video in additional formats.
        
        Args:
            frames: Frame array
            output_directory: Output directory
            base_filename: Base filename without extension
            fps: Frame rate
            export_formats: Comma-separated format list
            base_encoding_settings: Base encoding settings
            metadata: Metadata to embed
            
        Returns:
            Tuple of (additional_file_paths, total_additional_size_mb)
        """
        additional_files = []
        total_additional_size = 0.0
        
        # Parse export formats
        formats = [fmt.strip().lower() for fmt in export_formats.split(',') if fmt.strip()]
        
        # Remove extension from base filename if present
        base_name = os.path.splitext(base_filename)[0]
        
        for fmt in formats:
            try:
                # Skip if it's the same as primary format (case-insensitive)
                if fmt.lower() == 'mp4' and base_filename.lower().endswith('.mp4'):
                    continue
                
                # Create format-specific filename
                format_filename = f"{base_name}.{fmt}"
                format_path = os.path.join(output_directory, format_filename)
                format_path = os.path.abspath(format_path)
                
                # Adjust encoding settings for format
                format_settings = self._get_format_specific_settings(fmt, base_encoding_settings)
                
                print(f"Exporting additional format: {fmt}")
                
                # Encode in additional format
                format_stats = self._encode_video_ffmpeg_enhanced(
                    frames, format_path, fps, format_settings, metadata
                )
                
                additional_files.append(format_path)
                total_additional_size += format_stats['file_size_mb']
                
                print(f"Additional format {fmt} saved: {format_stats['file_size_mb']:.1f}MB")
                
            except Exception as e:
                print(f"Warning: Failed to export {fmt} format: {str(e)}")
                continue
        
        return additional_files, total_additional_size
    
    def _get_format_specific_settings(self, format_name: str, base_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get format-specific encoding settings.
        
        Args:
            format_name: Target format (e.g., 'webm', 'mov')
            base_settings: Base encoding settings
            
        Returns:
            Format-specific encoding settings
        """
        settings = base_settings.copy()
        
        # Format-specific codec mappings (case-insensitive)
        format_codecs = {
            'webm': 'libvpx-vp9',
            'mov': 'libx264',
            'avi': 'libx264',
            'mkv': 'libx264',
            'mp4': 'libx264'
        }
        
        # Override codec for specific formats (case-insensitive)
        format_lower = format_name.lower()
        if format_lower in format_codecs:
            settings['codec'] = format_codecs[format_lower]
            
            # Format-specific optimizations
            if format_name == 'webm':
                settings['pixel_format'] = 'yuv420p'
                settings['preset'] = 'medium'
                # VP9 uses different quality range
                if 'quality' in settings:
                    # Convert CRF from x264 range (0-51) to VP9 range (0-63)
                    settings['quality'] = min(63, int(settings['quality'] * 1.2))
            elif format_name in ['mov', 'mp4']:
                # Use original H.264 settings
                pass
        
        return settings
    
    def _sanitize_metadata_key(self, key: str) -> str:
        """
        Sanitize metadata key to prevent command injection.
        
        Args:
            key: Metadata key
            
        Returns:
            Sanitized key
        """
        # Allow only alphanumeric characters, underscores, and hyphens
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', key)
        
        # Limit length
        sanitized = sanitized[:50]
        
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'meta_' + sanitized
        
        return sanitized or 'metadata'
    
    def _sanitize_metadata_value(self, value: str) -> str:
        """
        Sanitize metadata value to prevent command injection.
        
        Args:
            value: Metadata value
            
        Returns:
            Sanitized value
        """
        # Remove potentially dangerous characters
        dangerous_chars = ['$', '`', '\\', ';', '|', '&', '>', '<', '"', "'"]
        
        sanitized = value
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Limit length to prevent buffer overflow
        sanitized = sanitized[:500]
        
        # Remove control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in ['\n', '\t'])
        
        return sanitized
    
    def _validate_codec_name(self, codec: str) -> str:
        """
        Validate codec name against whitelist to prevent injection.
        
        Args:
            codec: Codec name
            
        Returns:
            Validated codec name
            
        Raises:
            ValueError: If codec is not in whitelist
        """
        # Whitelist of allowed codecs
        allowed_codecs = {
            'libx264', 'libx265', 'libvpx', 'libvpx-vp9', 'prores_ks', 
            'mpeg4', 'libvorbis', 'aac', 'mp3', 'flac'
        }
        
        if codec not in allowed_codecs:
            print(f"Warning: Codec '{codec}' not in whitelist, using libx264")
            return 'libx264'
        
        return codec


# Required for ComfyUI node registration
WEB_DIRECTORY = "./web"