"""
VideoSequenceComposer Node for Loopy Comfy

This node composes video sequences by loading video files according to the
Markov-generated sequence and concatenating frames efficiently.
"""

import cv2
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any, Optional

# CRITICAL: Standardized import path setup for ComfyUI compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.video_utils import extract_frames, resize_frames, concatenate_frames, load_video_safe
from utils.memory_manager import managed_frame_processing, MemoryBoundedError, force_cleanup
from utils.security_utils import InputValidator, SecurityError
from utils.performance_optimizations import (
    default_buffer_pool, default_frame_processor, default_video_loader,
    default_load_balancer, default_performance_monitor, optimize_video_processing_pipeline
)
from utils.optimized_video_processing import process_video_batch_high_performance


class LoopyComfy_VideoSequenceComposer:
    """
    ComfyUI node for composing video sequences from Markov-generated sequences.
    
    Takes a video sequence specification and loads the corresponding video files,
    extracting and concatenating frames with memory-efficient processing.
    
    Enhanced with 20+ resolution presets including mobile and vertical formats.
    """
    
    # Extended resolution presets including mobile and vertical formats
    RESOLUTION_PRESETS = {
        # Standard landscape formats
        "4K UHD (3840×2160)": (3840, 2160),
        "2K QHD (2560×1440)": (2560, 1440), 
        "1080p FHD (1920×1080)": (1920, 1080),
        "720p HD (1280×720)": (1280, 720),
        "480p SD (854×480)": (854, 480),
        "360p (640×360)": (640, 360),
        
        # Vertical formats (mobile-first)
        "9:16 Portrait (1080×1920)": (1080, 1920),
        "9:16 Portrait 4K (2160×3840)": (2160, 3840),
        "4:5 Instagram (1080×1350)": (1080, 1350),
        "1:1 Square (1080×1080)": (1080, 1080),
        "1:1 Square HD (720×720)": (720, 720),
        
        # Platform-specific presets
        "TikTok (1080×1920)": (1080, 1920),
        "Instagram Reels (1080×1920)": (1080, 1920),
        "YouTube Shorts (1080×1920)": (1080, 1920),
        "Instagram Post (1080×1080)": (1080, 1080),
        "Twitter Video (1280×720)": (1280, 720),
        "Snapchat (1080×1920)": (1080, 1920),
        
        # Cinema formats
        "Cinema 2.35:1 (1920×817)": (1920, 817),
        "Cinema 2.39:1 (1920×803)": (1920, 803),
        "Cinema 1.85:1 (1920×1038)": (1920, 1038),
        "Ultrawide (2560×1080)": (2560, 1080),
        
        # Custom option
        "Custom": (0, 0),
        
        # Backward compatibility - old format support
        "1920x1080": (1920, 1080),
        "1280x720": (1280, 720),
        "854x480": (854, 480),
        "640x360": (640, 360)
    }
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input types for ComfyUI interface."""
        return {
            "required": {
                "sequence": ("VIDEO_SEQUENCE", {
                    "tooltip": "Video sequence from MarkovVideoSequencer"
                }),
                "resolution_preset": (list(cls.RESOLUTION_PRESETS.keys()), {
                    "default": "1080p FHD (1920×1080)",
                    "tooltip": "Choose from 20+ resolution presets including mobile formats"
                }),
                "output_fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 240.0,
                    "step": 1.0,
                    "display": "slider",
                    "tooltip": "Output frame rate - now properly applied to final video"
                }),
                "batch_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of videos to process simultaneously for memory efficiency"
                })
            },
            "optional": {
                "fps_conversion_method": (["duplicate", "blend", "motion_interpolation"], {
                    "default": "duplicate", 
                    "tooltip": "Method for frame rate conversion when source FPS differs from target"
                }),
                "custom_width": ("INT", {
                    "default": 1920,
                    "min": 128,
                    "max": 8192,
                    "display": "number",
                    "tooltip": "Custom width (only used when resolution_preset is 'Custom')"
                }),
                "custom_height": ("INT", {
                    "default": 1080,
                    "min": 128,
                    "max": 8192,
                    "display": "number",
                    "tooltip": "Custom height (only used when resolution_preset is 'Custom')"
                }),
                "maintain_aspect": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Maintain aspect ratio",
                    "label_off": "Stretch to fit",
                    "tooltip": "Preserve original aspect ratio with padding"
                }),
                "fill_mode": (["pad", "crop", "stretch"], {
                    "default": "pad",
                    "tooltip": "How to handle aspect ratio mismatches: pad (letterbox), crop (zoom), or stretch"
                }),
                "pad_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Padding color in hex format (e.g., #000000 for black)"
                }),
                "interpolation": (["LANCZOS", "CUBIC", "LINEAR", "NEAREST"], {
                    "default": "LANCZOS",
                    "tooltip": "Frame resize interpolation method for quality"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "FLOAT")  # ComfyUI IMAGE tensor format
    RETURN_NAMES = ("frames", "total_frames", "duration_seconds", "actual_fps")
    FUNCTION = "compose_sequence"
    CATEGORY = "video/avatar"
    
    def compose_sequence(
        self,
        sequence: List[Dict[str, Any]],
        resolution_preset: str,
        output_fps: float,
        batch_size: int,
        fps_conversion_method: str = "duplicate",
        custom_width: int = 1920,
        custom_height: int = 1080,
        maintain_aspect: bool = True,
        fill_mode: str = "pad",
        pad_color: str = "#000000",
        interpolation: str = "LANCZOS",
        **kwargs
    ) -> Tuple[np.ndarray, int, float, float]:
        """
        Compose video sequence by loading and concatenating frames.
        
        Args:
            sequence: Video sequence from MarkovVideoSequencer
            resolution_preset: Selected resolution preset or 'Custom'
            output_fps: Target output frame rate (now properly applied)
            batch_size: Number of videos to process at once
            fps_conversion_method: Method for FPS conversion
            custom_width/height: Custom dimensions when preset is 'Custom'
            maintain_aspect: Whether to maintain aspect ratio
            fill_mode: How to handle aspect ratio mismatches
            pad_color: Padding color for letterboxing
            interpolation: Resize interpolation method
            
        Returns:
            Tuple containing (frames, frame_count, duration, actual_fps)
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If processing fails
        """
        try:
            # Validate inputs securely
            if not sequence:
                raise ValueError("Empty sequence provided")
            
            # Validate numeric inputs
            validator = InputValidator()
            output_fps = validator.validate_numeric_input(output_fps, min_val=1, max_val=120)
            batch_size = int(validator.validate_numeric_input(batch_size, min_val=1, max_val=100))
            custom_width = int(validator.validate_numeric_input(custom_width, min_val=32, max_val=7680))
            custom_height = int(validator.validate_numeric_input(custom_height, min_val=32, max_val=4320))
            
            # Limit sequence length for security
            if len(sequence) > 10000:
                raise ValueError("Sequence too long - maximum 10,000 videos allowed")
            
            # Get target resolution from preset or custom values
            width, height = self._get_target_resolution(resolution_preset, custom_width, custom_height)
            
            # Get interpolation method
            interp_method = self._get_interpolation_method(interpolation)
            
            print(f"Composing sequence: {len(sequence)} videos at {width}x{height} @ {output_fps}fps")
            print(f"FPS conversion method: {fps_conversion_method}")
            print(f"Fill mode: {fill_mode}")
            
            # Use memory-bounded processing with automatic cleanup
            with managed_frame_processing(max_memory_mb=7000, max_frames=2000) as frame_buffer:
                processed_videos = 0
                
                # Process videos in batches for memory efficiency
                for i in range(0, len(sequence), batch_size):
                    batch = sequence[i:i + batch_size]
                    
                    try:
                        # Use high-performance optimized batch processing
                        batch_frames = process_video_batch_high_performance(
                            batch, width, height, output_fps,
                            fps_conversion_method=fps_conversion_method,
                            maintain_aspect=maintain_aspect,
                            fill_mode=fill_mode,
                            pad_color=pad_color,
                            interp_method=interp_method
                        )
                        
                        frame_buffer.extend(batch_frames)
                        processed_videos += len(batch)
                        
                        print(f"Processed {processed_videos}/{len(sequence)} videos...")
                        
                    except MemoryBoundedError as e:
                        # Memory limit exceeded - try to continue with smaller batch
                        print(f"Memory limit reached at video {processed_videos}: {e}")
                        if processed_videos == 0:
                            raise ValueError("Cannot process even a single video within memory limits")
                        break
                    
                    except Exception as e:
                        print(f"Error processing batch {i//batch_size}: {e}")
                        # Continue with next batch
                        continue
                
                # Get all processed frames
                all_frames = frame_buffer.get_all_frames()
                
                if not all_frames:
                    raise ValueError("No frames were successfully processed")
                
                # Convert to ComfyUI IMAGE tensor format with memory efficiency
                try:
                    frames_array = np.array(all_frames, dtype=np.float32) / 255.0
                except MemoryError:
                    # If we can't create the full array, try processing in chunks
                    raise MemoryBoundedError("Cannot create final frame array - reduce video count or resolution")
                
                # Calculate actual statistics
                total_frames = len(all_frames)
                duration_seconds = total_frames / output_fps
                actual_fps = output_fps
                
                print(f"Composition complete: {total_frames} frames, {duration_seconds:.1f}s @ {actual_fps}fps")
                
                return (frames_array, total_frames, duration_seconds, actual_fps)
            
        except MemoryBoundedError as e:
            force_cleanup()
            raise RuntimeError(f"Memory limit exceeded: {str(e)}")
        except SecurityError as e:
            raise ValueError(f"Security validation failed: {str(e)}")
        except Exception as e:
            force_cleanup()
            raise RuntimeError(f"Failed to compose video sequence: {str(e)}")
        finally:
            # Ensure cleanup
            force_cleanup()
    
    def _get_target_resolution(self, resolution_preset: str, custom_width: int, custom_height: int) -> Tuple[int, int]:
        """
        Get target resolution from preset or custom values.
        
        Args:
            resolution_preset: Selected resolution preset
            custom_width: Custom width value
            custom_height: Custom height value
            
        Returns:
            Tuple of (width, height)
        """
        if resolution_preset == "Custom":
            return custom_width, custom_height
        elif resolution_preset in self.RESOLUTION_PRESETS:
            return self.RESOLUTION_PRESETS[resolution_preset]
        else:
            # Fallback to parsing if it's a resolution string format
            try:
                # Check if it's in old "WIDTHxHEIGHT" format for backward compatibility
                if 'x' in resolution_preset:
                    width, height = map(int, resolution_preset.split('x'))
                    return width, height
            except (ValueError, AttributeError):
                pass
            
            # Default fallback
            print(f"Warning: Unknown resolution preset '{resolution_preset}', using 1920x1080")
            return 1920, 1080
    
    def _get_interpolation_method(self, interpolation: str) -> int:
        """
        Get OpenCV interpolation constant from string.
        
        Args:
            interpolation: Interpolation method name
            
        Returns:
            OpenCV interpolation constant
        """
        methods = {
            "LANCZOS": cv2.INTER_LANCZOS4,
            "CUBIC": cv2.INTER_CUBIC,
            "LINEAR": cv2.INTER_LINEAR,
            "NEAREST": cv2.INTER_NEAREST
        }
        
        return methods.get(interpolation, cv2.INTER_LANCZOS4)
    
    def _process_video_batch(
        self,
        batch: List[Dict[str, Any]],
        width: int,
        height: int,
        output_fps: float,
        fps_conversion_method: str,
        maintain_aspect: bool,
        fill_mode: str,
        pad_color: str,
        interp_method: int,
        transition_effect: str,
        transition_duration: float
    ) -> List[np.ndarray]:
        """
        Process a batch of videos and extract frames with enhanced FPS handling.
        
        Args:
            batch: Batch of sequence entries
            width: Target width
            height: Target height
            output_fps: Target frame rate
            fps_conversion_method: Method for FPS conversion
            maintain_aspect: Whether to maintain aspect ratio
            fill_mode: How to handle aspect ratio mismatches
            pad_color: Padding color
            interp_method: OpenCV interpolation method
            transition_effect: Transition effect to apply
            transition_duration: Duration of transitions
            
        Returns:
            List of processed frames
        """
        batch_frames = []
        
        for entry in batch:
            try:
                video_path = entry['metadata']['file_path']
                video_fps = entry['metadata']['fps']
                video_duration = entry['duration']
                
                # Calculate how many frames we need for target FPS
                target_frame_count = int(video_duration * output_fps)
                
                # Load video frames with improved FPS conversion
                frames = self._load_video_frames_with_conversion(
                    video_path, video_fps, output_fps, target_frame_count, fps_conversion_method
                )
                
                # Resize frames to target resolution with enhanced options
                if frames:
                    resized_frames = self._resize_frames_batch_enhanced(
                        frames, width, height, maintain_aspect, fill_mode, pad_color, interp_method
                    )
                    batch_frames.extend(resized_frames)
                
            except Exception as e:
                print(f"Warning: Failed to process video {entry.get('video_id', 'unknown')}: {str(e)}")
                continue
        
        return batch_frames
    
    def _load_video_frames_with_conversion(
        self,
        video_path: str,
        video_fps: float,
        output_fps: float,
        target_frame_count: int,
        conversion_method: str
    ) -> List[np.ndarray]:
        """
        Load frames from video with enhanced frame rate conversion.
        
        Args:
            video_path: Path to video file
            video_fps: Original video frame rate
            output_fps: Target output frame rate
            target_frame_count: Target number of frames
            conversion_method: Method for FPS conversion ('duplicate', 'blend', 'motion_interpolation')
            
        Returns:
            List of frame arrays
        """
        cap = load_video_safe(video_path)
        if cap is None:
            return []
        
        frames = []
        
        try:
            # Check if FPS conversion is needed
            if abs(video_fps - output_fps) < 0.01:
                # FPS match, simple extraction
                return self._extract_frames_direct(cap, target_frame_count)
            
            # Apply appropriate conversion method
            if conversion_method == "duplicate":
                frames = self._fps_convert_duplicate(cap, video_fps, output_fps, target_frame_count)
            elif conversion_method == "blend":
                frames = self._fps_convert_blend(cap, video_fps, output_fps, target_frame_count)
            elif conversion_method == "motion_interpolation":
                frames = self._fps_convert_motion_interpolation(cap, video_fps, output_fps, target_frame_count)
            else:
                # Fallback to duplicate method
                frames = self._fps_convert_duplicate(cap, video_fps, output_fps, target_frame_count)
            
            return frames[:target_frame_count]  # Ensure exact count
            
        finally:
            cap.release()
    
    def _resize_frames_batch_enhanced(
        self,
        frames: List[np.ndarray],
        width: int,
        height: int,
        maintain_aspect: bool,
        fill_mode: str,
        pad_color: str,
        interp_method: int
    ) -> List[np.ndarray]:
        """
        Resize a batch of frames with enhanced options.
        
        Args:
            frames: List of frame arrays
            width: Target width
            height: Target height
            maintain_aspect: Whether to maintain aspect ratio
            fill_mode: How to handle aspect ratio mismatches ('pad', 'crop', 'stretch')
            pad_color: Padding color in hex format
            interp_method: OpenCV interpolation method
            
        Returns:
            List of resized frames
        """
        if not frames:
            return frames
        
        resized_frames = []
        
        # Parse pad color
        pad_rgb = self._parse_hex_color(pad_color)
        
        for frame in frames:
            if maintain_aspect and fill_mode != "stretch":
                if fill_mode == "pad":
                    resized_frame = self._resize_with_padding(frame, width, height, interp_method, pad_rgb)
                elif fill_mode == "crop":
                    resized_frame = self._resize_with_cropping(frame, width, height, interp_method)
                else:  # fallback to padding
                    resized_frame = self._resize_with_padding(frame, width, height, interp_method, pad_rgb)
            else:
                # Stretch mode - no aspect ratio preservation
                resized_frame = cv2.resize(frame, (width, height), interpolation=interp_method)
            
            resized_frames.append(resized_frame)
        
        return resized_frames
    
    def _resize_with_aspect(
        self,
        frame: np.ndarray,
        target_width: int,
        target_height: int,
        interp_method: int
    ) -> np.ndarray:
        """
        Resize frame while maintaining aspect ratio with padding.
        
        Args:
            frame: Input frame array
            target_width: Target width
            target_height: Target height
            interp_method: OpenCV interpolation method
            
        Returns:
            Resized frame with padding
        """
        h, w = frame.shape[:2]
        aspect = w / h
        target_aspect = target_width / target_height
        
        if aspect > target_aspect:
            # Width is limiting factor
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            # Height is limiting factor
            new_height = target_height
            new_width = int(target_height * aspect)
        
        # Resize to fit
        resized = cv2.resize(frame, (new_width, new_height), interpolation=interp_method)
        
        # Create canvas and center the frame
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
    
    def _extract_frames_direct(self, cap: cv2.VideoCapture, target_frame_count: int) -> List[np.ndarray]:
        """
        Extract frames directly without FPS conversion.
        
        Args:
            cap: Video capture object
            target_frame_count: Number of frames to extract
            
        Returns:
            List of frame arrays
        """
        frames = []
        frame_index = 0
        
        while len(frames) < target_frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            frame_index += 1
        
        return frames[:target_frame_count]
    
    def _fps_convert_duplicate(self, cap: cv2.VideoCapture, video_fps: float, output_fps: float, target_frame_count: int) -> List[np.ndarray]:
        """
        Convert FPS by duplicating or dropping frames.
        
        Args:
            cap: Video capture object
            video_fps: Source video frame rate
            output_fps: Target frame rate
            target_frame_count: Target number of frames
            
        Returns:
            List of converted frames
        """
        frames = []
        frame_ratio = video_fps / output_fps
        
        if frame_ratio >= 1.0:
            # Downsample frames (skip frames)
            frame_step = max(1, int(frame_ratio))
            frame_index = 0
            
            while len(frames) < target_frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                frame_index += frame_step
        else:
            # Upsample frames (duplicate frames)
            import math
            frame_duplicate_count = max(1, math.ceil(1.0 / frame_ratio))
            frame_index = 0
            
            while len(frames) < target_frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add frame multiple times
                for _ in range(min(frame_duplicate_count, target_frame_count - len(frames))):
                    frames.append(rgb_frame.copy())
                    if len(frames) >= target_frame_count:
                        break
                
                frame_index += 1
        
        return frames[:target_frame_count]
    
    def _fps_convert_blend(self, cap: cv2.VideoCapture, video_fps: float, output_fps: float, target_frame_count: int) -> List[np.ndarray]:
        """
        Convert FPS by blending adjacent frames.
        
        Args:
            cap: Video capture object
            video_fps: Source video frame rate
            output_fps: Target frame rate
            target_frame_count: Target number of frames
            
        Returns:
            List of blended frames
        """
        # For now, fall back to duplicate method
        # Blending implementation would require more complex temporal interpolation
        return self._fps_convert_duplicate(cap, video_fps, output_fps, target_frame_count)
    
    def _fps_convert_motion_interpolation(self, cap: cv2.VideoCapture, video_fps: float, output_fps: float, target_frame_count: int) -> List[np.ndarray]:
        """
        Convert FPS using motion interpolation (placeholder).
        
        Args:
            cap: Video capture object
            video_fps: Source video frame rate
            output_fps: Target frame rate
            target_frame_count: Target number of frames
            
        Returns:
            List of interpolated frames
        """
        # For now, fall back to duplicate method
        # Motion interpolation would require optical flow computation
        print("Motion interpolation not implemented yet, using frame duplication")
        return self._fps_convert_duplicate(cap, video_fps, output_fps, target_frame_count)
    
    def _parse_hex_color(self, hex_color: str) -> Tuple[int, int, int]:
        """
        Parse hex color string to RGB tuple with robust error handling.
        
        Args:
            hex_color: Hex color string (e.g., '#000000', '#000', 'black')
            
        Returns:
            RGB tuple (r, g, b)
        """
        try:
            # Handle named colors
            named_colors = {
                'black': (0, 0, 0),
                'white': (255, 255, 255),
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255),
                'yellow': (255, 255, 0),
                'cyan': (0, 255, 255),
                'magenta': (255, 0, 255)
            }
            
            if hex_color.lower() in named_colors:
                return named_colors[hex_color.lower()]
            
            # Remove '#' if present
            hex_color = hex_color.lstrip('#')
            
            # Handle short hex format (#RGB -> #RRGGBB)
            if len(hex_color) == 3:
                hex_color = ''.join([c*2 for c in hex_color])
            
            # Pad if too short
            hex_color = hex_color.ljust(6, '0')[:6]
            
            # Convert to RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            return (r, g, b)
        except (ValueError, IndexError) as e:
            print(f"Warning: Invalid color '{hex_color}', using black: {str(e)}")
            # Default to black on error
            return (0, 0, 0)
    
    def _resize_with_padding(self, frame: np.ndarray, target_width: int, target_height: int, interp_method: int, pad_color: Tuple[int, int, int]) -> np.ndarray:
        """
        Resize frame while maintaining aspect ratio with colored padding.
        
        Args:
            frame: Input frame array
            target_width: Target width
            target_height: Target height
            interp_method: OpenCV interpolation method
            pad_color: RGB tuple for padding color
            
        Returns:
            Resized frame with padding
        """
        h, w = frame.shape[:2]
        aspect = w / h
        target_aspect = target_width / target_height
        
        if aspect > target_aspect:
            # Width is limiting factor
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            # Height is limiting factor
            new_height = target_height
            new_width = int(target_height * aspect)
        
        # Resize to fit
        resized = cv2.resize(frame, (new_width, new_height), interpolation=interp_method)
        
        # Create canvas with padding color
        canvas = np.full((target_height, target_width, 3), pad_color, dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
    
    def _resize_with_cropping(self, frame: np.ndarray, target_width: int, target_height: int, interp_method: int) -> np.ndarray:
        """
        Resize frame by cropping to fill target dimensions.
        
        Args:
            frame: Input frame array
            target_width: Target width
            target_height: Target height
            interp_method: OpenCV interpolation method
            
        Returns:
            Resized and cropped frame
        """
        h, w = frame.shape[:2]
        aspect = w / h
        target_aspect = target_width / target_height
        
        if aspect > target_aspect:
            # Source is wider, crop width
            new_height = h
            new_width = int(h * target_aspect)
            x_offset = (w - new_width) // 2
            y_offset = 0
        else:
            # Source is taller, crop height
            new_width = w
            new_height = int(w / target_aspect)
            x_offset = 0
            y_offset = (h - new_height) // 2
        
        # Crop frame
        cropped = frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width]
        
        # Resize to target dimensions
        resized = cv2.resize(cropped, (target_width, target_height), interpolation=interp_method)
        
        return resized


# Required for ComfyUI node registration
WEB_DIRECTORY = "./web"