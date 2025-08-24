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

# FIXED: Import security utilities with fallback for ComfyUI compatibility
try:
    from utils.security_utils import PathValidator, SecurityError, default_resource_limiter
    SECURITY_AVAILABLE = True
except ImportError as e:
    # Fallback implementations for when security utilities can't be imported
    SECURITY_AVAILABLE = False
    
    class SecurityError(Exception):
        """Fallback SecurityError."""
        pass
    
    class PathValidator:
        """Fallback PathValidator with basic validation."""
        def __init__(self, allowed_base_dirs=None):
            self.allowed_base_dirs = allowed_base_dirs or []
        
        def validate_directory_path(self, directory_path: str) -> str:
            """Basic path validation fallback."""
            if not directory_path or not isinstance(directory_path, str):
                raise SecurityError("Invalid directory path")
            
            # Basic normalization
            abs_path = os.path.abspath(directory_path)
            if not os.path.exists(abs_path):
                raise SecurityError(f"Directory does not exist: {abs_path}")
            
            return abs_path
    
    class ResourceLimiter:
        """Fallback ResourceLimiter."""
        def check_memory_usage(self):
            return True
        
        def with_memory_limit(self, func, *args, **kwargs):
            return func(*args, **kwargs)
    
    default_resource_limiter = ResourceLimiter()


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
                    "dynamicPrompts": False,
                    "tooltip": "Video directory path. Use the Browse button for native OS dialog."
                }),
                "browse_folder": (["📁 Browse Folder"], {
                    "default": "📁 Browse Folder",
                    "tooltip": "Open native OS folder selection dialog"
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
                }),
                "show_preview": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Show Video Preview",
                    "label_off": "No Preview",
                    "tooltip": "Generate 360p preview (max 150 frames, memory safe)"
                })
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
    
    RETURN_TYPES = ("VIDEO_METADATA_LIST", "INT", "FLOAT", "IMAGE", "STRING")
    RETURN_NAMES = ("video_metadata", "video_count", "total_duration", "preview_grid", "folder_info")
    FUNCTION = "load_video_assets"
    CATEGORY = "video/avatar"
    
    def load_video_assets(
        self, 
        directory_path: str, 
        browse_folder: str = "📁 Browse Folder",
        file_pattern: str = "*.mp4", 
        max_videos: int = 100,
        validate_seamless: bool = True,
        scan_subfolders: bool = False,
        sort_by: str = "name",
        filter_resolution: str = "all",
        preview_mode: str = "list",
        show_preview: bool = False,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], int, float, Optional[np.ndarray], str]:
        """
        Load and validate video assets from directory.
        
        Args:
            directory_path: Path to directory containing video files
            browse_folder: Folder browser button (triggers dialog when activated)
            file_pattern: Glob pattern for filtering files
            max_videos: Maximum number of videos to load
            validate_seamless: Whether to validate seamless loop points
            show_preview: Whether to generate memory-safe preview
            
        Returns:
            Tuple containing (video_metadata, count, duration, preview, folder_info)
            
        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If no valid videos found
        """
        try:
            # CRITICAL: Handle folder browser button activation
            if browse_folder == "📁 Browse Folder" and TKINTER_AVAILABLE:
                # Trigger folder dialog and update directory_path if selection made
                dialog_result = self.open_folder_dialog(directory_path)
                if dialog_result and dialog_result.get('success') and dialog_result.get('directory_path'):
                    directory_path = dialog_result['directory_path']
                    print(f"Selected folder via dialog: {directory_path}")
            # Validate directory path with security checks
            directory_path = self._validate_directory_path(directory_path)
            file_pattern = self._validate_file_pattern(file_pattern)
            
            # Check resource limits to prevent DoS attacks
            if not default_resource_limiter.check_memory_usage():
                raise ValueError("Memory limit exceeded - reduce batch size or free memory")
            
            if max_videos > 1000:  # Prevent excessive resource usage
                max_videos = 1000
                print("WARNING: max_videos limited to 1000 for security reasons")
            
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
            
            # Generate preview grid based on mode and settings
            preview_grid = None
            if show_preview:
                preview_grid = self._generate_memory_safe_preview(video_metadata, preview_mode)
            else:
                preview_grid = self._generate_preview_grid(video_metadata, preview_mode)
            
            # Create folder information string
            folder_info = f"Loaded {video_count} videos from {directory_path} (Total: {total_duration:.1f}s)"
            
            print(f"Successfully loaded {video_count} video assets")
            print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
            
            return (video_metadata, video_count, total_duration, preview_grid, folder_info)
            
        except Exception as e:
            # Return safe defaults on error to maintain compatibility
            error_msg = f"Error loading assets: {str(e)}"
            print(f"ERROR: {error_msg}")
            return ([], 0, 0.0, None, error_msg)
    
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
    
    def open_folder_dialog(self, current_path: str = None) -> Optional[Dict[str, str]]:
        """
        Open native OS folder selection dialog.
        
        Args:
            current_path: Current directory path to use as initial directory
        
        Returns:
            Dictionary with selected folder path, video count, and additional info
        """
        if not TKINTER_AVAILABLE:
            print("Warning: Folder dialog not available (tkinter not installed)")
            return {
                "success": False, 
                "error": "Tkinter not available",
                "fallback_suggestions": [
                    "./assets/videos/",
                    "./input/", 
                    "./ComfyUI/input/",
                    "C:\\Users\\%USERNAME%\\Videos",
                    "/home/%USER%/Videos"
                ]
            }
        
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            root.attributes('-topmost', True)  # Bring dialog to front
            root.lift()  # Additional lift for focus
            
            # Use current path or reasonable default as initial directory
            initial_dir = current_path if current_path and os.path.exists(current_path) else os.getcwd()
            
            folder_path = filedialog.askdirectory(
                title="Select Video Folder - LoopyComfy",
                initialdir=initial_dir,
                mustexist=True
            )
            
            root.destroy()
            
            if folder_path and os.path.exists(folder_path):
                # Quick scan to count videos in selected folder
                try:
                    video_extensions = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv', '.wmv'}
                    video_files = [f for f in os.listdir(folder_path) 
                                 if os.path.splitext(f)[1].lower() in video_extensions]
                    video_count = len(video_files)
                    
                    return {
                        "success": True,
                        "directory_path": folder_path,
                        "path": folder_path,  # For compatibility
                        "video_count": video_count,
                        "folder_name": os.path.basename(folder_path)
                    }
                except Exception as scan_error:
                    print(f"Warning: Could not scan selected folder: {scan_error}")
                    return {
                        "success": True,
                        "directory_path": folder_path,
                        "path": folder_path,
                        "video_count": "unknown",
                        "folder_name": os.path.basename(folder_path)
                    }
            
            return {"success": False, "cancelled": True}
            
        except Exception as e:
            print(f"Error opening folder dialog: {str(e)}")
            return {
                "success": False, 
                "error": str(e),
                "fallback_suggestions": [
                    "./assets/videos/",
                    "./input/",
                    "./ComfyUI/input/"
                ]
            }
    
    def _generate_memory_safe_preview(self, video_metadata: List[Dict[str, Any]], preview_mode: str) -> Optional[np.ndarray]:
        """
        Generate memory-safe preview with strict constraints.
        
        CRITICAL: Must not exceed 1GB additional memory usage
        - Maximum 150 frames (5 seconds at 30fps)
        - 360p resolution (640x360) for memory efficiency 
        - Maximum 10 videos for preview
        
        Args:
            video_metadata: List of video metadata dictionaries
            preview_mode: Preview display mode
            
        Returns:
            Memory-safe preview image array or None
        """
        if not video_metadata or preview_mode == "list":
            return None
            
        try:
            # STEP 1: Memory safety constraints
            MAX_PREVIEW_VIDEOS = min(10, len(video_metadata))
            MAX_FRAMES = 150  # 5 seconds at 30fps
            PREVIEW_WIDTH = 640
            PREVIEW_HEIGHT = 360
            
            # Estimate memory usage: frames * width * height * channels * bytes
            estimated_memory_mb = (MAX_FRAMES * PREVIEW_WIDTH * PREVIEW_HEIGHT * 3 * MAX_PREVIEW_VIDEOS) / (1024 * 1024)
            
            if estimated_memory_mb > 1000:  # 1GB limit
                print(f"Warning: Preview would use {estimated_memory_mb:.1f}MB, skipping for memory safety")
                return None
            
            print(f"Generating memory-safe preview: {MAX_PREVIEW_VIDEOS} videos, max {MAX_FRAMES} frames each")
            print(f"Estimated memory usage: {estimated_memory_mb:.1f}MB")
            
            # STEP 2: Load and resize frames from sample videos
            preview_frames = []
            videos_processed = 0
            
            for metadata in video_metadata[:MAX_PREVIEW_VIDEOS]:
                if videos_processed >= MAX_PREVIEW_VIDEOS:
                    break
                    
                try:
                    cap = cv2.VideoCapture(metadata['file_path'])
                    if not cap.isOpened():
                        continue
                    
                    # Sample frames evenly throughout video
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_step = max(1, total_frames // min(MAX_FRAMES // MAX_PREVIEW_VIDEOS, total_frames))
                    
                    video_frames = []
                    frame_count = 0
                    current_frame = 0
                    
                    while frame_count < (MAX_FRAMES // MAX_PREVIEW_VIDEOS) and current_frame < total_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                        ret, frame = cap.read()
                        
                        if ret:
                            # Resize to 360p for memory efficiency
                            frame_resized = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                            video_frames.append(frame_rgb)
                            frame_count += 1
                        
                        current_frame += frame_step
                    
                    cap.release()
                    
                    if video_frames:
                        preview_frames.extend(video_frames)
                        videos_processed += 1
                    
                    # Memory check during processing
                    current_memory_mb = len(preview_frames) * PREVIEW_WIDTH * PREVIEW_HEIGHT * 3 / (1024 * 1024)
                    if current_memory_mb > 1000:
                        print(f"Memory limit reached at {current_memory_mb:.1f}MB, stopping preview generation")
                        break
                        
                except Exception as e:
                    print(f"Warning: Failed to process {metadata.get('filename', 'unknown')} for preview: {str(e)}")
                    continue
            
            if not preview_frames:
                print("No frames could be loaded for preview")
                return None
            
            # STEP 3: Create final preview array
            # Convert to numpy array and normalize
            preview_array = np.array(preview_frames, dtype=np.float32) / 255.0
            
            # Add batch dimension for ComfyUI compatibility
            if len(preview_array.shape) == 4:  # (frames, height, width, channels)
                preview_array = np.expand_dims(preview_array, axis=0)  # (1, frames, height, width, channels)
                
            final_memory_mb = preview_array.nbytes / (1024 * 1024)
            print(f"Preview generated: {len(preview_frames)} frames, {final_memory_mb:.1f}MB memory used")
            
            return preview_array
            
        except Exception as e:
            print(f"Error generating memory-safe preview: {str(e)}")
            return None
    
    def _validate_directory_path(self, directory_path: str) -> str:
        """
        Secure validation and sanitization of directory paths.
        
        Args:
            directory_path: Input directory path
            
        Returns:
            Validated absolute path
            
        Raises:
            ValueError: If path contains malicious patterns or is not allowed
        """
        try:
            path_validator = PathValidator()
            return path_validator.validate_directory_path(directory_path)
        except SecurityError as e:
            raise ValueError(f"Security validation failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Path validation error: {str(e)}")
    
    def _validate_file_pattern(self, file_pattern: str) -> str:
        """
        Secure validation and sanitization of file patterns.
        
        Args:
            file_pattern: Input file pattern
            
        Returns:
            Validated file pattern
            
        Raises:
            ValueError: If pattern contains dangerous characters or patterns
        """
        try:
            path_validator = PathValidator()
            return path_validator.validate_file_pattern(file_pattern)
        except SecurityError as e:
            raise ValueError(f"File pattern security validation failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"File pattern validation error: {str(e)}")


# Required for ComfyUI node registration
WEB_DIRECTORY = "./web"