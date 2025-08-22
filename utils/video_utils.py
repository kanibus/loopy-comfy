"""
Video Processing Utilities

Comprehensive utilities for video loading, frame processing, and codec operations
used throughout the NonLinear Video Avatar system.
"""

import os
import cv2
import numpy as np
import ffmpeg
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


def load_video_safe(video_path: str) -> Optional[cv2.VideoCapture]:
    """
    Safely load a video file with comprehensive error handling.
    
    Args:
        video_path: Path to video file
        
    Returns:
        OpenCV VideoCapture object or None if failed
    """
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        return cap
        
    except Exception as e:
        print(f"Error loading video {video_path}: {str(e)}")
        return None


def extract_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary containing video metadata
        
    Raises:
        ValueError: If video cannot be processed
    """
    cap = load_video_safe(video_path)
    if cap is None:
        raise ValueError(f"Could not load video for metadata extraction: {video_path}")
    
    try:
        # Basic properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Additional properties
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        # File information
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
            "codec": codec,
            "file_size": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "bitrate": (file_size * 8) / duration if duration > 0 else 0
        }
        
        return metadata
        
    finally:
        cap.release()


def validate_seamless(video_path: str, threshold: float = 100.0) -> bool:
    """
    Validate that a video forms a seamless loop.
    
    Args:
        video_path: Path to video file
        threshold: MSE threshold for seamless detection
        
    Returns:
        True if video appears to be seamless, False otherwise
    """
    cap = load_video_safe(video_path)
    if cap is None:
        return False
    
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
        
        # Compare frames
        mse = calculate_frame_mse(first_frame, last_frame)
        return mse < threshold
        
    finally:
        cap.release()


def calculate_frame_mse(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate Mean Squared Error between two frames.
    
    Args:
        frame1: First frame array
        frame2: Second frame array
        
    Returns:
        MSE value
    """
    # Convert to grayscale for comparison
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate MSE
    mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
    return mse


def convert_bgr_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert BGR frame to RGB format.
    
    Args:
        frame: BGR frame array
        
    Returns:
        RGB frame array
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Extract all frames from a video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of frame arrays in RGB format
        
    Raises:
        ValueError: If video cannot be processed
    """
    cap = load_video_safe(video_path)
    if cap is None:
        raise ValueError(f"Could not load video: {video_path}")
    
    frames = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = convert_bgr_rgb(frame)
            frames.append(rgb_frame)
            
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
        
        return frames
        
    finally:
        cap.release()


def resize_frames(
    frames: List[np.ndarray], 
    target_width: int, 
    target_height: int,
    maintain_aspect: bool = True
) -> List[np.ndarray]:
    """
    Resize frames to target dimensions.
    
    Args:
        frames: List of frame arrays
        target_width: Target width
        target_height: Target height
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        List of resized frames
    """
    if not frames:
        return frames
    
    resized_frames = []
    
    for frame in frames:
        if maintain_aspect:
            # Calculate aspect ratio preserving dimensions
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
            
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Pad to target dimensions if needed
            if new_width != target_width or new_height != target_height:
                # Create black canvas
                canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                
                # Center the resized frame
                y_offset = (target_height - new_height) // 2
                x_offset = (target_width - new_width) // 2
                
                canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
                resized = canvas
                
        else:
            # Direct resize without aspect ratio preservation
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        resized_frames.append(resized)
    
    return resized_frames


def concatenate_frames(frame_lists: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Efficiently concatenate multiple lists of frames.
    
    Args:
        frame_lists: List of frame lists to concatenate
        
    Returns:
        Single concatenated list of frames
    """
    if not frame_lists:
        return []
    
    # Use list comprehension for efficiency
    concatenated = []
    for frame_list in frame_lists:
        concatenated.extend(frame_list)
    
    return concatenated


def check_codec_availability(codec: str) -> bool:
    """
    Check if a specific codec is available on the system.
    
    Args:
        codec: Codec name (e.g., 'libx264', 'libx265')
        
    Returns:
        True if codec is available, False otherwise
    """
    try:
        # Use ffmpeg-python to check codec availability
        probe = ffmpeg.probe('dummy', f=f'null', codec=codec, t=0.1, loglevel='quiet')
        return True
    except:
        return False


def get_optimal_codec(prefer_quality: bool = True) -> str:
    """
    Get the optimal codec available on the system.
    
    Args:
        prefer_quality: If True, prefer quality over compression
        
    Returns:
        Best available codec name
    """
    if prefer_quality:
        codecs = ['libx264', 'libx265', 'mpeg4']
    else:
        codecs = ['libx265', 'libx264', 'mpeg4']
    
    for codec in codecs:
        if check_codec_availability(codec):
            return codec
    
    # Fallback to default
    return 'libx264'


def estimate_file_size(
    frame_count: int, 
    width: int, 
    height: int, 
    fps: float,
    codec: str = 'libx264',
    quality: int = 23
) -> int:
    """
    Estimate output file size for given parameters.
    
    Args:
        frame_count: Number of frames
        width: Video width
        height: Video height
        fps: Frames per second
        codec: Video codec
        quality: Quality setting (CRF for H.264/H.265)
        
    Returns:
        Estimated file size in bytes
    """
    # Rough estimation based on typical compression ratios
    pixel_count = width * height * frame_count
    
    # Compression factors by codec and quality
    compression_factors = {
        'libx264': {18: 0.8, 23: 0.4, 28: 0.2},
        'libx265': {18: 0.6, 23: 0.3, 28: 0.15},
        'mpeg4': {18: 1.0, 23: 0.5, 28: 0.25}
    }
    
    # Get compression factor (default to medium quality)
    codec_factors = compression_factors.get(codec, compression_factors['libx264'])
    compression_factor = codec_factors.get(quality, 0.4)
    
    # Base calculation: 3 bytes per pixel (RGB) * compression factor
    estimated_size = int(pixel_count * 3 * compression_factor)
    
    return estimated_size


def get_video_info_summary(metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics for a collection of video metadata.
    
    Args:
        metadata_list: List of video metadata dictionaries
        
    Returns:
        Summary statistics dictionary
    """
    if not metadata_list:
        return {}
    
    durations = [m['duration'] for m in metadata_list]
    file_sizes = [m['file_size'] for m in metadata_list]
    resolutions = [f"{m['width']}x{m['height']}" for m in metadata_list]
    
    summary = {
        "total_videos": len(metadata_list),
        "total_duration": sum(durations),
        "average_duration": np.mean(durations),
        "min_duration": min(durations),
        "max_duration": max(durations),
        "total_file_size": sum(file_sizes),
        "total_file_size_mb": sum(file_sizes) / (1024 * 1024),
        "unique_resolutions": list(set(resolutions)),
        "seamless_count": sum(1 for m in metadata_list if m.get('is_seamless', False))
    }
    
    return summary