"""
VideoAssetLoader Node for Loopy Comfy

This node loads and validates video assets from a directory, extracting metadata
and preparing them for Markov chain sequencing.
"""

import os
import glob
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class LoopyComfy_VideoAssetLoader:
    """
    ComfyUI node for loading and validating video assets for avatar generation.
    
    Scans a directory for video files, validates them as seamless loops,
    and extracts metadata required for Markov chain sequencing.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input types for ComfyUI interface."""
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": "./assets/videos/",
                    "multiline": False,
                    "placeholder": "Path to video directory"
                }),
                "file_pattern": ("STRING", {
                    "default": "*.mp4",
                    "multiline": False,
                    "placeholder": "File pattern (e.g., *.mp4, avatar_*.mov)"
                }),
                "max_videos": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
                "validate_seamless": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Validate loops",
                    "label_off": "Skip validation"
                })
            }
        }
    
    RETURN_TYPES = ("VIDEO_METADATA_LIST",)
    RETURN_NAMES = ("video_metadata",)
    FUNCTION = "load_video_assets"
    CATEGORY = "video/avatar"
    
    def load_video_assets(
        self, 
        directory_path: str, 
        file_pattern: str, 
        max_videos: int,
        validate_seamless: bool
    ) -> Tuple[List[Dict[str, Any]]]:
        """
        Load and validate video assets from directory.
        
        Args:
            directory_path: Path to directory containing video files
            file_pattern: Glob pattern for filtering files
            max_videos: Maximum number of videos to load
            validate_seamless: Whether to validate seamless loop points
            
        Returns:
            Tuple containing list of video metadata dictionaries
            
        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If no valid videos found
        """
        try:
            # Validate directory path
            if not os.path.exists(directory_path):
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            if not os.path.isdir(directory_path):
                raise ValueError(f"Path is not a directory: {directory_path}")
            
            # Find video files matching pattern
            search_pattern = os.path.join(directory_path, file_pattern)
            video_files = glob.glob(search_pattern)
            
            if not video_files:
                raise ValueError(f"No video files found matching pattern: {search_pattern}")
            
            # Limit number of videos
            video_files = sorted(video_files)[:max_videos]
            
            video_metadata = []
            for video_path in video_files:
                try:
                    metadata = self._extract_video_metadata(video_path, validate_seamless)
                    video_metadata.append(metadata)
                except Exception as e:
                    print(f"Warning: Skipping {video_path}: {str(e)}")
                    continue
            
            if not video_metadata:
                raise ValueError("No valid video files could be processed")
            
            print(f"Successfully loaded {len(video_metadata)} video assets")
            return (video_metadata,)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load video assets: {str(e)}")
    
    def _extract_video_metadata(
        self, 
        video_path: str, 
        validate_seamless: bool
    ) -> Dict[str, Any]:
        """
        Extract metadata from a single video file.
        
        Args:
            video_path: Path to video file
            validate_seamless: Whether to validate seamless looping
            
        Returns:
            Dictionary containing video metadata
            
        Raises:
            ValueError: If video is invalid or not seamless when validation enabled
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            # Extract basic metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Validate basic properties
            if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
                raise ValueError("Invalid video properties")
            
            # Validate seamless loop if requested
            is_seamless = True
            if validate_seamless:
                is_seamless = self._validate_seamless_loop(cap)
                if not is_seamless:
                    raise ValueError("Video is not a seamless loop")
            
            # Get file stats
            file_stats = os.stat(video_path)
            file_size = file_stats.st_size
            
            metadata = {
                "file_path": os.path.abspath(video_path),
                "filename": os.path.basename(video_path),
                "duration": duration,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "file_size": file_size,
                "is_seamless": is_seamless,
                "video_id": os.path.splitext(os.path.basename(video_path))[0]
            }
            
            return metadata
            
        finally:
            cap.release()
    
    def _validate_seamless_loop(self, cap: cv2.VideoCapture) -> bool:
        """
        Validate that video forms a seamless loop by comparing first and last frames.
        
        Args:
            cap: OpenCV VideoCapture object
            
        Returns:
            True if video appears to be a seamless loop, False otherwise
        """
        try:
            # Get first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, first_frame = cap.read()
            if not ret:
                return False
            
            # Get last frame
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ret, last_frame = cap.read()
            if not ret:
                return False
            
            # Compare frames using structural similarity
            # Convert to grayscale for comparison
            gray1 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate mean squared error
            mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
            
            # Threshold for seamless detection (adjustable)
            # Lower MSE indicates more similar frames
            seamless_threshold = 100.0  # Can be tuned based on testing
            
            return mse < seamless_threshold
            
        except Exception:
            # If validation fails, assume not seamless
            return False


# Required for ComfyUI node registration
WEB_DIRECTORY = "./web"