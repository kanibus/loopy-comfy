"""
Tests for VideoSequenceComposer Node - ComfyUI NonLinear Video Avatar
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add project path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.video_composer import LoopyComfy_VideoSequenceComposer
from .conftest import create_test_video


class TestVideoSequenceComposer:
    """Test suite for VideoSequenceComposer ComfyUI node."""
    
    @pytest.fixture
    def node(self):
        """Create VideoSequenceComposer node instance."""
        return LoopyComfy_VideoSequenceComposer()
    
    @pytest.fixture
    def sample_sequence(self, temp_video_dir):
        """Create sample video sequence with actual video files."""
        video_files = []
        
        # Create test videos
        for i in range(3):
            video_path = os.path.join(temp_video_dir, f"test_video_{i}.mp4")
            create_test_video(video_path, duration=2.0, fps=30.0)
            video_files.append(video_path)
        
        # Create sequence data structure
        sequence = [
            {
                "video_id": "test_video_0",
                "file_path": video_files[0],
                "filename": "test_video_0.mp4",
                "duration": 2.0,
                "start_time": 0.0,
                "end_time": 2.0
            },
            {
                "video_id": "test_video_1", 
                "file_path": video_files[1],
                "filename": "test_video_1.mp4",
                "duration": 2.0,
                "start_time": 2.0,
                "end_time": 4.0
            },
            {
                "video_id": "test_video_2",
                "file_path": video_files[2],
                "filename": "test_video_2.mp4",
                "duration": 2.0,
                "start_time": 4.0,
                "end_time": 6.0
            }
        ]
        
        return sequence
    
    def test_input_types_structure(self, node):
        """Test INPUT_TYPES returns correct structure."""
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        required = input_types["required"]
        
        # Check required fields
        assert "sequence" in required
        assert "output_fps" in required
        assert "resolution" in required
        assert "batch_size" in required
        
        # Check optional fields
        assert "optional" in input_types
        optional = input_types["optional"]
        assert "memory_limit_gb" in optional
        assert "frame_interpolation" in optional
    
    def test_return_types(self, node):
        """Test RETURN_TYPES returns correct types."""
        return_types = node.RETURN_TYPES
        assert return_types == ("IMAGE", "STRING")
    
    def test_return_names(self, node):
        """Test RETURN_NAMES returns correct names."""
        return_names = node.RETURN_NAMES
        assert return_names == ("frames", "status_message")
    
    def test_function_name(self, node):
        """Test FUNCTION returns correct function name."""
        function_name = node.FUNCTION
        assert function_name == "compose_sequence"
    
    def test_category(self, node):
        """Test CATEGORY returns correct category."""
        category = node.CATEGORY
        assert category == "video/avatar"
    
    def test_compose_sequence_basic(self, node, sample_sequence):
        """Test basic sequence composition."""
        frames, status = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=10
        )
        
        assert frames is not None
        assert isinstance(frames, np.ndarray)
        assert "Successfully composed" in status
        
        # Check frame tensor shape (B, H, W, C)
        batch_size, height, width, channels = frames.shape
        assert height == 480
        assert width == 640
        assert channels == 3
        
        # Should have frames for all videos (2 seconds each at 30fps = 60 frames each)
        expected_frames = 3 * 60  # 3 videos * 60 frames each
        assert batch_size >= expected_frames * 0.9  # Allow some tolerance
    
    def test_compose_sequence_different_resolution(self, node, sample_sequence):
        """Test composition with different resolution."""
        frames, status = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="1280x720",
            batch_size=20
        )
        
        assert frames is not None
        batch_size, height, width, channels = frames.shape
        assert height == 720
        assert width == 1280
        assert channels == 3
    
    def test_compose_sequence_different_fps(self, node, sample_sequence):
        """Test composition with different frame rate."""
        frames_30fps, _ = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=10
        )
        
        frames_15fps, _ = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=15.0,
            resolution="640x480",
            batch_size=10
        )
        
        # 15fps should have roughly half the frames of 30fps
        assert frames_15fps.shape[0] < frames_30fps.shape[0]
        assert frames_15fps.shape[0] >= frames_30fps.shape[0] * 0.4  # Allow tolerance
    
    def test_memory_limit_handling(self, node, sample_sequence):
        """Test memory limit enforcement."""
        frames, status = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=5,  # Small batch
            memory_limit_gb=1.0  # 1GB limit
        )
        
        assert frames is not None
        assert "memory" in status.lower() or "composed" in status.lower()
    
    def test_batch_processing(self, node, sample_sequence):
        """Test that batch processing works correctly."""
        # Test with small batch size
        frames_small, _ = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=2  # Very small batch
        )
        
        # Test with large batch size
        frames_large, _ = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=50  # Large batch
        )
        
        # Results should be similar regardless of batch size
        assert frames_small.shape == frames_large.shape
        
        # Check that frames are similar (allowing for minor differences)
        diff = np.mean(np.abs(frames_small.astype(float) - frames_large.astype(float)))
        assert diff < 1.0  # Very small difference allowed
    
    def test_empty_sequence(self, node):
        """Test error handling with empty sequence."""
        empty_sequence = []
        
        frames, status = node.compose_sequence(
            sequence=empty_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=10
        )
        
        assert frames is None or frames.shape[0] == 0
        assert "Error" in status or "empty" in status.lower()
    
    def test_invalid_video_files(self, node):
        """Test handling of invalid video file paths."""
        invalid_sequence = [
            {
                "video_id": "nonexistent",
                "file_path": "/nonexistent/path/video.mp4",
                "filename": "video.mp4",
                "duration": 2.0,
                "start_time": 0.0,
                "end_time": 2.0
            }
        ]
        
        frames, status = node.compose_sequence(
            sequence=invalid_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=10
        )
        
        # Should handle gracefully
        assert "Error" in status or "failed" in status.lower()
    
    def test_frame_interpolation_enabled(self, node, sample_sequence):
        """Test frame interpolation feature."""
        frames, status = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=60.0,  # Higher than source (30fps)
            resolution="640x480",
            batch_size=10,
            frame_interpolation=True
        )
        
        if frames is not None:
            # Should have more frames due to interpolation
            batch_size = frames.shape[0]
            expected_min_frames = 3 * 2 * 60  # 3 videos * 2 seconds * 60 fps
            assert batch_size >= expected_min_frames * 0.8  # Allow tolerance
    
    def test_frame_interpolation_disabled(self, node, sample_sequence):
        """Test without frame interpolation."""
        frames, status = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=60.0,  # Higher than source
            resolution="640x480",
            batch_size=10,
            frame_interpolation=False
        )
        
        if frames is not None:
            # Without interpolation, should duplicate frames instead
            batch_size = frames.shape[0]
            # Still expect higher frame count, but through duplication
            assert batch_size > 0
    
    @patch('utils.video_utils.load_video_safe')
    def test_video_loading_error_handling(self, mock_load_video, node, sample_sequence):
        """Test error handling when video loading fails."""
        # Mock video loading to fail
        mock_load_video.return_value = None
        
        frames, status = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=10
        )
        
        assert "Error" in status or "failed" in status.lower()
    
    @patch('utils.video_utils.extract_frames')
    def test_frame_extraction_error_handling(self, mock_extract_frames, node, sample_sequence):
        """Test error handling when frame extraction fails."""
        # Mock frame extraction to fail
        mock_extract_frames.return_value = []
        
        frames, status = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=10
        )
        
        # Should handle gracefully
        assert "Error" in status or "composed" in status.lower()
    
    def test_resolution_parsing(self, node):
        """Test resolution string parsing."""
        test_cases = [
            ("1920x1080", (1080, 1920)),
            ("1280x720", (720, 1280)),
            ("640x480", (480, 640)),
            ("854x480", (480, 854))
        ]
        
        for res_str, expected in test_cases:
            height, width = node.parse_resolution(res_str)
            assert (height, width) == expected
    
    def test_invalid_resolution_format(self, node, sample_sequence):
        """Test handling of invalid resolution format."""
        frames, status = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="invalid_format",
            batch_size=10
        )
        
        # Should either use default resolution or return error
        if frames is not None:
            # If successful, should use default resolution
            assert len(frames.shape) == 4
        else:
            assert "Error" in status
    
    def test_memory_usage_monitoring(self, node, sample_sequence):
        """Test that memory usage is monitored."""
        import tracemalloc
        tracemalloc.start()
        
        frames, status = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=10,
            memory_limit_gb=8.0
        )
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Should not exceed reasonable memory usage
        if frames is not None:
            # Memory usage should be proportional to frame data
            frame_memory = frames.nbytes if hasattr(frames, 'nbytes') else 0
            assert peak < frame_memory * 3  # Allow 3x overhead
    
    def test_frame_data_type_and_range(self, node, sample_sequence):
        """Test frame data type and value range."""
        frames, status = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=10
        )
        
        if frames is not None:
            # Check data type (should be uint8 or float32)
            assert frames.dtype in [np.uint8, np.float32]
            
            # Check value range
            if frames.dtype == np.uint8:
                assert frames.min() >= 0
                assert frames.max() <= 255
            elif frames.dtype == np.float32:
                assert frames.min() >= 0.0
                assert frames.max() <= 1.0
    
    def test_sequence_continuity(self, node, sample_sequence):
        """Test that sequence maintains temporal continuity."""
        frames, status = node.compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=10
        )
        
        if frames is not None and frames.shape[0] > 1:
            # Check that consecutive frames are reasonably similar
            # (indicating smooth transitions, not random jumps)
            frame_diffs = []
            for i in range(min(10, frames.shape[0] - 1)):
                diff = np.mean(np.abs(
                    frames[i].astype(float) - frames[i+1].astype(float)
                ))
                frame_diffs.append(diff)
            
            # Average frame difference should be reasonable
            avg_diff = np.mean(frame_diffs)
            assert avg_diff < 50.0  # Adjust threshold as needed
    
    def test_large_sequence_handling(self, node, temp_video_dir):
        """Test handling of large sequences."""
        # Create larger sequence
        large_sequence = []
        for i in range(10):  # 10 videos
            video_path = os.path.join(temp_video_dir, f"large_test_{i}.mp4")
            create_test_video(video_path, duration=1.0, fps=30.0)
            
            large_sequence.append({
                "video_id": f"large_test_{i}",
                "file_path": video_path,
                "filename": f"large_test_{i}.mp4",
                "duration": 1.0,
                "start_time": float(i),
                "end_time": float(i + 1)
            })
        
        frames, status = node.compose_sequence(
            sequence=large_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=5,  # Small batch for memory efficiency
            memory_limit_gb=2.0
        )
        
        # Should handle large sequence without crashing
        if frames is not None:
            expected_frames = 10 * 30  # 10 videos * 1 second * 30 fps
            assert frames.shape[0] >= expected_frames * 0.8  # Allow tolerance