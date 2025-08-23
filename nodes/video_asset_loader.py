# -*- coding: utf-8 -*-
"""
VideoAssetLoader Node for Loopy Comfy

This node loads and validates video assets from a directory, extracting metadata
and preparing them for Markov chain sequencing.
"""

import os
import glob
import sys
import cv2
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any, Optional
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# CRITICAL: Standardized import path setup for ComfyUI compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


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
                    "placeholder": "Path to video directory",
                    "dynamicPrompts": False
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
                    "display": "number",
                    "tooltip": "Maximum number of videos to load from directory"
                }),
                "validate_seamless": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Validate loops",
                    "label_off": "Skip validation",
                    "tooltip": "Check if videos form seamless loops by comparing first/last frames"
                })
            },
            "optional": {
                "scan_subfolders": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Include Subfolders",
                    "label_off": "Current Folder Only",
                    "tooltip": "Recursively scan subdirectories for video files"
                }),
                "sort_by": (["name", "date", "size", "duration"], {
                    "default": "name",
                    "tooltip": "Sort videos by specified criteria"
                }),
                "filter_resolution": (["all", "1080p", "720p", "4K", "vertical", "square"], {
                    "default": "all",
                    "tooltip": "Filter videos by resolution characteristics"
                }),
                "preview_mode": (["thumbnails", "list", "details"], {
                    "default": "list",
                    "tooltip": "Display mode for loaded video information"
                })
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
    
    RETURN_TYPES = ("VIDEO_METADATA_LIST", "INT", "FLOAT", "IMAGE")
    RETURN_NAMES = ("video_metadata", "video_count", "total_duration", "preview_grid")
    FUNCTION = "load_video_assets"
    CATEGORY = "video/avatar"
    
    def load_video_assets(
        self, 
        directory_path: str, 
        file_pattern: str, 
        max_videos: int,
        validate_seamless: bool,
        scan_subfolders: bool = False,
        sort_by: str = "name",
        filter_resolution: str = "all",
        preview_mode: str = "list",
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], int, float, Optional[np.ndarray]]:
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
            # Validate directory path with security checks
            directory_path = self._validate_directory_path(directory_path)
            file_pattern = self._validate_file_pattern(file_pattern)
            
            if not os.path.exists(directory_path):
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            if not os.path.isdir(directory_path):
                raise ValueError(f"Path is not a directory: {directory_path}")
            
            # Find video files matching pattern with optional recursive scanning
            video_files = self._find_video_files(directory_path, file_pattern, scan_subfolders)
            
            if not video_files:
                raise ValueError(f"No video files found matching pattern: {file_pattern} in {directory_path}")
            
            # Sort files according to sort_by preference
            video_files = self._sort_video_files(video_files, sort_by)
            
            # Limit number of videos
            video_files = video_files[:max_videos]
            
            video_metadata = []
            for i, video_path in enumerate(video_files):
                try:
                    metadata = self._extract_video_metadata(video_path, validate_seamless)
                    video_metadata.append(metadata)
                    
                    # Progress reporting for large collections
                    if i % 10 == 0 and i > 0:
                        print(f"Processing videos: {i}/{len(video_files)}...")
                        
                except Exception as e:
                    print(f"Warning: Skipping {video_path}: {str(e)}")
                    continue
            
            if not video_metadata:
                raise ValueError("No valid video files could be processed")
            
            # Filter by resolution if specified
            if filter_resolution != "all":
                video_metadata = self._filter_by_resolution(video_metadata, filter_resolution)
                
            if not video_metadata:
                raise ValueError(f"No videos found matching resolution filter: {filter_resolution}")
            
            # Calculate statistics
            video_count = len(video_metadata)
            total_duration = sum(metadata['duration'] for metadata in video_metadata)
            
            # Generate preview grid based on mode
            preview_grid = self._generate_preview_grid(video_metadata, preview_mode)
            
            print(f"Successfully loaded {video_count} video assets")
            print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
            
            return (video_metadata, video_count, total_duration, preview_grid)
            
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
    
    def _find_video_files(self, directory_path: str, file_pattern: str, scan_subfolders: bool) -> List[str]:
        """
        Find video files matching pattern, optionally scanning subdirectories.
        
        Args:
            directory_path: Base directory to search
            file_pattern: File pattern to match
            scan_subfolders: Whether to scan recursively
            
        Returns:
            List of video file paths
        """
        video_files = []
        
        if scan_subfolders:
            # Recursive search
            for root, dirs, files in os.walk(directory_path):
                search_pattern = os.path.join(root, file_pattern)
                video_files.extend(glob.glob(search_pattern))
        else:
            # Single directory search
            search_pattern = os.path.join(directory_path, file_pattern)
            video_files = glob.glob(search_pattern)
        
        # Filter to only include common video formats
        video_extensions = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv', '.wmv'}
        video_files = [f for f in video_files if os.path.splitext(f)[1].lower() in video_extensions]
        
        return video_files
    
    def _sort_video_files(self, video_files: List[str], sort_by: str) -> List[str]:
        """
        Sort video files by specified criteria.
        
        Args:
            video_files: List of video file paths
            sort_by: Sorting criteria ('name', 'date', 'size', 'duration')
            
        Returns:
            Sorted list of video file paths
        """
        if sort_by == "name":
            return sorted(video_files, key=lambda f: os.path.basename(f).lower())
        elif sort_by == "date":
            return sorted(video_files, key=lambda f: os.path.getmtime(f), reverse=True)
        elif sort_by == "size":
            return sorted(video_files, key=lambda f: os.path.getsize(f), reverse=True)
        elif sort_by == "duration":
            # Sort by duration requires metadata extraction (expensive)
            # For now, fall back to name sorting
            print("Duration sorting not implemented yet, using name sorting")
            return sorted(video_files, key=lambda f: os.path.basename(f).lower())
        else:
            return video_files
    
    def _filter_by_resolution(self, video_metadata: List[Dict[str, Any]], filter_type: str) -> List[Dict[str, Any]]:
        """
        Filter video metadata by resolution characteristics.
        
        Args:
            video_metadata: List of video metadata dictionaries
            filter_type: Filter type ('1080p', '720p', '4K', 'vertical', 'square')
            
        Returns:
            Filtered list of video metadata
        """
        filtered_videos = []
        
        for metadata in video_metadata:
            width = metadata['width']
            height = metadata['height']
            
            # Safe aspect ratio calculation
            if height > 0:
                aspect_ratio = width / height
            else:
                print(f"Warning: Invalid height ({height}) for video {metadata.get('filename', 'unknown')}")
                continue
            
            if filter_type == "1080p" and height == 1080:
                filtered_videos.append(metadata)
            elif filter_type == "720p" and height == 720:
                filtered_videos.append(metadata)
            elif filter_type == "4K" and height >= 2160:
                filtered_videos.append(metadata)
            elif filter_type == "vertical" and aspect_ratio < 1.0:
                filtered_videos.append(metadata)
            elif filter_type == "square" and 0.95 <= aspect_ratio <= 1.05:
                filtered_videos.append(metadata)
        
        return filtered_videos
    
    def _generate_preview_grid(self, video_metadata: List[Dict[str, Any]], preview_mode: str) -> Optional[np.ndarray]:
        """
        Generate preview visualization of loaded videos.
        
        Args:
            video_metadata: List of video metadata dictionaries
            preview_mode: Preview display mode
            
        Returns:
            Preview image array or None
        """
        try:
            if preview_mode == "thumbnails":
                return self._create_thumbnail_grid(video_metadata)
            elif preview_mode == "details":
                return self._create_details_view(video_metadata)
            else:  # list mode
                return self._create_list_view(video_metadata)
        except Exception as e:
            print(f"Warning: Failed to generate preview: {str(e)}")
            return None
    
    def _create_thumbnail_grid(self, video_metadata: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """
        Create thumbnail grid showing first frame of each video.
        
        Args:
            video_metadata: List of video metadata dictionaries
            
        Returns:
            Thumbnail grid image array
        """
        if not video_metadata:
            return None
            
        # Limit to reasonable number for performance
        max_thumbnails = min(16, len(video_metadata))
        
        thumbnails = []
        thumb_size = (120, 68)  # 16:9 aspect ratio
        
        for i, metadata in enumerate(video_metadata[:max_thumbnails]):
            try:
                cap = cv2.VideoCapture(metadata['file_path'])
                try:
                    ret, frame = cap.read()
                    
                    if ret:
                        # Resize and convert to RGB
                        thumbnail = cv2.resize(frame, thumb_size)
                        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                        thumbnails.append(thumbnail)
                    else:
                        # Create placeholder thumbnail
                        placeholder = np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
                        thumbnails.append(placeholder)
                finally:
                    cap.release()  # Ensure always released
            except Exception as e:
                print(f"Warning: Failed to generate thumbnail for {metadata.get('filename', 'unknown')}: {str(e)}")
                # Create placeholder for failed videos
                placeholder = np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
                thumbnails.append(placeholder)
        
        if not thumbnails:
            return None
        
        # Arrange thumbnails in grid
        grid_cols = min(4, len(thumbnails))
        grid_rows = (len(thumbnails) + grid_cols - 1) // grid_cols
        
        # Create grid canvas
        grid_width = grid_cols * thumb_size[0]
        grid_height = grid_rows * thumb_size[1]
        grid_canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for i, thumb in enumerate(thumbnails):
            row = i // grid_cols
            col = i % grid_cols
            y_start = row * thumb_size[1]
            y_end = y_start + thumb_size[1]
            x_start = col * thumb_size[0]
            x_end = x_start + thumb_size[0]
            
            grid_canvas[y_start:y_end, x_start:x_end] = thumb
        
        # Convert to ComfyUI format (add batch dimension and normalize)
        grid_canvas = grid_canvas.astype(np.float32) / 255.0
        grid_canvas = np.expand_dims(grid_canvas, axis=0)  # Add batch dimension
        
        return grid_canvas
    
    def _create_list_view(self, video_metadata: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """
        Create simple text-based list view (placeholder implementation).
        
        Args:
            video_metadata: List of video metadata dictionaries
            
        Returns:
            None (text list would be handled by UI)
        """
        # For now, return None as text display would be handled by ComfyUI interface
        # In future, could render text to image
        return None
    
    def _create_details_view(self, video_metadata: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """
        Create detailed view with statistics visualization.
        
        Args:
            video_metadata: List of video metadata dictionaries
            
        Returns:
            Details visualization image array
        """
        # Placeholder for future implementation
        # Could create charts showing resolution distribution, duration statistics, etc.
        return None
    
    def open_folder_dialog(self) -> Optional[Dict[str, str]]:
        """
        Open native OS folder selection dialog.
        
        Returns:
            Dictionary with selected folder path or None
        """
        if not TKINTER_AVAILABLE:
            print("Warning: Folder dialog not available (tkinter not installed)")
            return None
        
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            root.attributes('-topmost', True)  # Bring dialog to front
            
            folder_path = filedialog.askdirectory(
                title="Select Video Folder",
                initialdir=os.getcwd()
            )
            
            root.destroy()
            
            if folder_path:
                return {"directory_path": folder_path}
            return None
            
        except Exception as e:
            print(f"Error opening folder dialog: {str(e)}")
            return None
    
    def _validate_directory_path(self, directory_path: str) -> str:
        """
        Validate and sanitize directory path to prevent path traversal attacks.
        
        Args:
            directory_path: Input directory path
            
        Returns:
            Sanitized absolute path
            
        Raises:
            ValueError: If path contains malicious patterns
        """
        # Convert to absolute path
        abs_path = os.path.abspath(directory_path)
        
        # Check for path traversal patterns
        dangerous_patterns = ['..', '~', '$']
        normalized_path = os.path.normpath(directory_path.lower())
        
        for pattern in dangerous_patterns:
            if pattern in normalized_path:
                raise ValueError(f"Potentially unsafe path detected: {directory_path}")
        
        # Ensure path doesn't escape from reasonable bounds
        # Allow only paths that are subdirectories of current working directory or absolute paths
        cwd = os.path.abspath(os.getcwd())
        
        # If it's not under current directory and not an explicit absolute path, reject it
        if not (abs_path.startswith(cwd) or os.path.isabs(directory_path)):
            raise ValueError(f"Path not allowed: {directory_path}")
        
        # Additional safety: check for Windows drive letter issues (Windows specific)
        if os.name == 'nt':  # Windows
            if len(abs_path) > 260:  # Windows MAX_PATH limitation
                raise ValueError(f"Path too long (>{260} chars): {directory_path}")
        
        return abs_path
    
    def _validate_file_pattern(self, file_pattern: str) -> str:
        """
        Validate and sanitize file pattern to prevent shell injection.
        
        Args:
            file_pattern: Input file pattern
            
        Returns:
            Sanitized file pattern
            
        Raises:
            ValueError: If pattern contains dangerous characters
        """
        # Remove potentially dangerous characters
        dangerous_chars = [';', '|', '&', '>', '<', '`', '$', '(', ')']
        
        for char in dangerous_chars:
            if char in file_pattern:
                raise ValueError(f"Dangerous character '{char}' not allowed in file pattern: {file_pattern}")
        
        # Ensure pattern looks like a valid glob pattern
        if not any(c in file_pattern for c in ['*', '?', '[', ']']):
            if not file_pattern.startswith('*.'):
                # If it's not a glob pattern, make it one
                if '.' in file_pattern:
                    file_pattern = '*' + file_pattern
                else:
                    file_pattern = file_pattern + '*'
        
        return file_pattern


# Required for ComfyUI node registration
WEB_DIRECTORY = "./web"