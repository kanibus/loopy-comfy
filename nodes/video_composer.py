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


class LoopyComfy_VideoSequenceComposer:
    """
    ComfyUI node for composing video sequences from Markov-generated sequences.
    
    Takes a video sequence specification and loads the corresponding video files,
    extracting and concatenating frames with memory-efficient processing.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input types for ComfyUI interface."""
        return {
            "required": {
                "sequence": ("VIDEO_SEQUENCE", {
                    "tooltip": "Video sequence from MarkovVideoSequencer"
                }),
                "output_fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Output frame rate"
                }),
                "resolution": (["1920x1080", "1280x720", "854x480", "640x360"], {
                    "default": "1920x1080",
                    "tooltip": "Output resolution"
                }),
                "batch_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of videos to process simultaneously"
                })
            },
            "optional": {
                "maintain_aspect": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Maintain aspect ratio",
                    "label_off": "Stretch to fit",
                    "tooltip": "Preserve original aspect ratio with padding"
                }),
                "interpolation": (["LANCZOS", "CUBIC", "LINEAR", "NEAREST"], {
                    "default": "LANCZOS",
                    "tooltip": "Frame resize interpolation method"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)  # ComfyUI IMAGE tensor format
    RETURN_NAMES = ("frames",)
    FUNCTION = "compose_sequence"
    CATEGORY = "video/avatar"
    
    def compose_sequence(
        self,
        sequence: List[Dict[str, Any]],
        output_fps: float,
        resolution: str,
        batch_size: int,
        maintain_aspect: bool = True,
        interpolation: str = "LANCZOS"
    ) -> Tuple[np.ndarray]:
        """
        Compose video sequence by loading and concatenating frames.
        
        Args:
            sequence: Video sequence from MarkovVideoSequencer
            output_fps: Target output frame rate
            resolution: Target output resolution
            batch_size: Number of videos to process at once
            maintain_aspect: Whether to maintain aspect ratio
            interpolation: Resize interpolation method
            
        Returns:
            Tuple containing concatenated frames as ComfyUI IMAGE tensor
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If processing fails
        """
        try:
            # Validate inputs
            if not sequence:
                raise ValueError("Empty sequence provided")
            
            if output_fps <= 0:
                raise ValueError("Output FPS must be positive")
            
            # Parse resolution
            width, height = self._parse_resolution(resolution)
            
            # Get interpolation method
            interp_method = self._get_interpolation_method(interpolation)
            
            print(f"Composing sequence: {len(sequence)} videos at {resolution} @ {output_fps}fps")
            
            all_frames = []
            processed_videos = 0
            
            # Process videos in batches for memory efficiency
            for i in range(0, len(sequence), batch_size):
                batch = sequence[i:i + batch_size]
                batch_frames = self._process_video_batch(
                    batch, width, height, output_fps, maintain_aspect, interp_method
                )
                
                all_frames.extend(batch_frames)
                processed_videos += len(batch)
                
                print(f"Processed {processed_videos}/{len(sequence)} videos...")
            
            if not all_frames:
                raise ValueError("No frames were successfully processed")
            
            # Convert to ComfyUI IMAGE tensor format
            # ComfyUI expects shape: (batch, height, width, channels) with values 0-1
            frames_array = np.array(all_frames, dtype=np.float32) / 255.0
            
            print(f"Composition complete: {len(all_frames)} frames, "
                  f"{len(all_frames)/output_fps:.1f}s duration")
            
            return (frames_array,)
            
        except Exception as e:
            raise RuntimeError(f"Failed to compose video sequence: {str(e)}")
    
    def _parse_resolution(self, resolution: str) -> Tuple[int, int]:
        """
        Parse resolution string into width and height.
        
        Args:
            resolution: Resolution string (e.g., "1920x1080")
            
        Returns:
            Tuple of (width, height)
        """
        try:
            width, height = map(int, resolution.split('x'))
            return width, height
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid resolution format: {resolution}")
    
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
        maintain_aspect: bool,
        interp_method: int
    ) -> List[np.ndarray]:
        """
        Process a batch of videos and extract frames.
        
        Args:
            batch: Batch of sequence entries
            width: Target width
            height: Target height
            output_fps: Target frame rate
            maintain_aspect: Whether to maintain aspect ratio
            interp_method: OpenCV interpolation method
            
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
                
                # Load video frames
                frames = self._load_video_frames(
                    video_path, video_fps, output_fps, target_frame_count
                )
                
                # Resize frames to target resolution
                if frames:
                    resized_frames = self._resize_frames_batch(
                        frames, width, height, maintain_aspect, interp_method
                    )
                    batch_frames.extend(resized_frames)
                
            except Exception as e:
                print(f"Warning: Failed to process video {entry.get('video_id', 'unknown')}: {str(e)}")
                continue
        
        return batch_frames
    
    def _load_video_frames(
        self,
        video_path: str,
        video_fps: float,
        output_fps: float,
        target_frame_count: int
    ) -> List[np.ndarray]:
        """
        Load frames from video with frame rate adaptation.
        
        Args:
            video_path: Path to video file
            video_fps: Original video frame rate
            output_fps: Target output frame rate
            target_frame_count: Target number of frames
            
        Returns:
            List of frame arrays
        """
        cap = load_video_safe(video_path)
        if cap is None:
            return []
        
        frames = []
        
        try:
            # Calculate frame sampling strategy
            frame_ratio = video_fps / output_fps
            
            if frame_ratio >= 1.0:
                # Downsample frames (skip frames)
                frame_step = int(frame_ratio)
                frame_index = 0
                
                while len(frames) < target_frame_count:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)
                    
                    frame_index += frame_step
            else:
                # Upsample frames (duplicate frames)
                frame_duplicate_count = int(1.0 / frame_ratio)
                
                frame_index = 0
                while len(frames) < target_frame_count:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Add frame multiple times
                    for _ in range(min(frame_duplicate_count, target_frame_count - len(frames))):
                        frames.append(rgb_frame.copy())
                        if len(frames) >= target_frame_count:
                            break
                    
                    frame_index += 1
            
            return frames[:target_frame_count]  # Ensure exact count
            
        finally:
            cap.release()
    
    def _resize_frames_batch(
        self,
        frames: List[np.ndarray],
        width: int,
        height: int,
        maintain_aspect: bool,
        interp_method: int
    ) -> List[np.ndarray]:
        """
        Resize a batch of frames efficiently.
        
        Args:
            frames: List of frame arrays
            width: Target width
            height: Target height
            maintain_aspect: Whether to maintain aspect ratio
            interp_method: OpenCV interpolation method
            
        Returns:
            List of resized frames
        """
        if not frames:
            return frames
        
        resized_frames = []
        
        for frame in frames:
            if maintain_aspect:
                resized_frame = self._resize_with_aspect(frame, width, height, interp_method)
            else:
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


# Required for ComfyUI node registration
WEB_DIRECTORY = "./web"