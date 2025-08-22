"""
Test configuration and fixtures for NonLinear Video Avatar tests.
"""

import pytest
import tempfile
import os
import numpy as np
import cv2
from typing import List, Dict, Any


@pytest.fixture
def temp_video_dir():
    """Create a temporary directory for test videos."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_video_metadata():
    """Sample video metadata for testing."""
    return [
        {
            "video_id": "test_video_001",
            "file_path": "/fake/path/test_video_001.mp4",
            "filename": "test_video_001.mp4",
            "duration": 5.0,
            "fps": 30.0,
            "frame_count": 150,
            "width": 1920,
            "height": 1080,
            "resolution": "1920x1080",
            "file_size": 1024000,
            "is_seamless": True
        },
        {
            "video_id": "test_video_002", 
            "file_path": "/fake/path/test_video_002.mp4",
            "filename": "test_video_002.mp4",
            "duration": 4.5,
            "fps": 30.0,
            "frame_count": 135,
            "width": 1920,
            "height": 1080,
            "resolution": "1920x1080",
            "file_size": 980000,
            "is_seamless": True
        },
        {
            "video_id": "test_video_003",
            "file_path": "/fake/path/test_video_003.mp4", 
            "filename": "test_video_003.mp4",
            "duration": 5.5,
            "fps": 30.0,
            "frame_count": 165,
            "width": 1920,
            "height": 1080,
            "resolution": "1920x1080",
            "file_size": 1100000,
            "is_seamless": True
        }
    ]


def create_test_video(output_path: str, duration: float = 5.0, fps: float = 30.0, 
                     width: int = 640, height: int = 480, seamless: bool = True) -> str:
    """
    Create a test video file for testing purposes.
    
    Args:
        output_path: Path where to save the test video
        duration: Duration in seconds
        fps: Frame rate
        width: Video width
        height: Video height
        seamless: Whether to make the video seamless (first/last frame similar)
        
    Returns:
        Path to created video file
    """
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = int(duration * fps)
    
    try:
        for i in range(frame_count):
            # Create a simple test pattern
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            if seamless:
                # Use sinusoidal pattern that loops seamlessly
                phase = (i / frame_count) * 2 * np.pi
                color_value = int((np.sin(phase) + 1) * 127)
            else:
                # Linear gradient that doesn't loop
                color_value = int((i / frame_count) * 255)
            
            # Fill frame with pattern
            frame[:, :] = [color_value, color_value // 2, 255 - color_value]
            
            # Add frame number text
            cv2.putText(frame, f"Frame {i}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
    
    finally:
        out.release()
    
    return output_path


@pytest.fixture 
def create_test_videos(temp_video_dir):
    """Create multiple test videos in temporary directory."""
    video_paths = []
    
    for i in range(3):
        video_path = os.path.join(temp_video_dir, f"test_video_{i:03d}.mp4")
        create_test_video(
            video_path, 
            duration=5.0 + i * 0.5,  # Varying durations
            seamless=True
        )
        video_paths.append(video_path)
    
    return video_paths


@pytest.fixture
def mock_comfyui_environment(monkeypatch):
    """Mock ComfyUI environment for testing."""
    # Mock any ComfyUI-specific imports or globals if needed
    pass