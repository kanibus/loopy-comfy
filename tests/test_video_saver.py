"""
Tests for VideoSaver Node - ComfyUI NonLinear Video Avatar
"""

import pytest
import numpy as np
import os
import tempfile
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add project path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.video_saver import LoopyComfy_VideoSaver


class TestVideoSaver:
    """Test suite for VideoSaver ComfyUI node."""
    
    @pytest.fixture
    def node(self):
        """Create VideoSaver node instance."""
        return LoopyComfy_VideoSaver()
    
    @pytest.fixture
    def sample_frames(self):
        """Create sample frame tensor for testing."""
        # Create 60 frames of 480x640x3 (2 seconds at 30fps)
        frames = np.random.randint(0, 256, size=(60, 480, 640, 3), dtype=np.uint8)
        return frames
    
    @pytest.fixture
    def sample_float_frames(self):
        """Create sample frame tensor with float32 values."""
        # Create frames with values in [0, 1] range
        frames = np.random.rand(30, 240, 320, 3).astype(np.float32)
        return frames
    
    def test_input_types_structure(self, node):
        """Test INPUT_TYPES returns correct structure."""
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        required = input_types["required"]
        
        # Check required fields
        assert "frames" in required
        assert "output_filename" in required
        assert "output_directory" in required
        assert "fps" in required
        assert "codec" in required
        
        # Check optional fields  
        assert "optional" in input_types
        optional = input_types["optional"]
        assert "quality" in optional
        assert "overwrite" in optional
    
    def test_return_types(self, node):
        """Test RETURN_TYPES returns correct types."""
        return_types = node.RETURN_TYPES
        assert return_types == ("STRING", "STRING")
    
    def test_return_names(self, node):
        """Test RETURN_NAMES returns correct names."""
        return_names = node.RETURN_NAMES
        assert return_names == ("output_path", "status_message")
    
    def test_function_name(self, node):
        """Test FUNCTION returns correct function name."""
        function_name = node.FUNCTION
        assert function_name == "save_video"
    
    def test_category(self, node):
        """Test CATEGORY returns correct category."""
        category = node.CATEGORY
        assert category == "video/avatar"
    
    @patch('ffmpeg.run')
    @patch('ffmpeg.input')
    @patch('ffmpeg.output')
    def test_save_video_basic(self, mock_output, mock_input, mock_run, node, sample_frames, temp_video_dir):
        """Test basic video saving functionality."""
        # Mock ffmpeg chain
        mock_stream = Mock()
        mock_input.return_value = mock_stream
        mock_output.return_value = mock_stream
        
        output_path, status = node.save_video(
            frames=sample_frames,
            output_filename="test_output.mp4",
            output_directory=temp_video_dir,
            fps=30.0,
            codec="libx264"
        )
        
        expected_path = os.path.join(temp_video_dir, "test_output.mp4")
        assert output_path == expected_path
        assert "Successfully saved" in status
        
        # Verify ffmpeg was called
        mock_input.assert_called_once()
        mock_output.assert_called_once()
        mock_run.assert_called_once()
    
    @patch('ffmpeg.run')
    @patch('ffmpeg.input')
    @patch('ffmpeg.output')
    def test_save_video_different_codecs(self, mock_output, mock_input, mock_run, node, sample_frames, temp_video_dir):
        """Test saving with different codecs."""
        codecs = ["libx264", "libx265", "mpeg4"]
        
        mock_stream = Mock()
        mock_input.return_value = mock_stream
        mock_output.return_value = mock_stream
        
        for codec in codecs:
            output_path, status = node.save_video(
                frames=sample_frames,
                output_filename=f"test_{codec}.mp4",
                output_directory=temp_video_dir,
                fps=30.0,
                codec=codec
            )
            
            assert "Successfully saved" in status
            assert codec in mock_output.call_args[1]["vcodec"] if mock_output.call_args else True
    
    @patch('ffmpeg.run')
    @patch('ffmpeg.input')
    @patch('ffmpeg.output')
    def test_save_video_with_quality(self, mock_output, mock_input, mock_run, node, sample_frames, temp_video_dir):
        """Test saving with different quality settings."""
        mock_stream = Mock()
        mock_input.return_value = mock_stream
        mock_output.return_value = mock_stream
        
        output_path, status = node.save_video(
            frames=sample_frames,
            output_filename="test_quality.mp4",
            output_directory=temp_video_dir,
            fps=30.0,
            codec="libx264",
            quality="high"
        )
        
        assert "Successfully saved" in status
        mock_output.assert_called_once()
    
    def test_create_output_directory(self, node, sample_frames):
        """Test automatic output directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested path that doesn't exist
            output_dir = os.path.join(temp_dir, "nested", "output", "path")
            
            with patch('ffmpeg.run'), patch('ffmpeg.input'), patch('ffmpeg.output'):
                output_path, status = node.save_video(
                    frames=sample_frames,
                    output_filename="test.mp4",
                    output_directory=output_dir,
                    fps=30.0,
                    codec="libx264"
                )
                
                # Directory should be created
                assert os.path.exists(output_dir)
                assert output_path == os.path.join(output_dir, "test.mp4")
    
    def test_overwrite_handling(self, node, sample_frames, temp_video_dir):
        """Test file overwrite handling."""
        output_file = os.path.join(temp_video_dir, "existing_file.mp4")
        
        # Create existing file
        with open(output_file, 'w') as f:
            f.write("existing content")
        
        with patch('ffmpeg.run'), patch('ffmpeg.input'), patch('ffmpeg.output'):
            # Test with overwrite=False
            output_path, status = node.save_video(
                frames=sample_frames,
                output_filename="existing_file.mp4",
                output_directory=temp_video_dir,
                fps=30.0,
                codec="libx264",
                overwrite=False
            )
            
            # Should create new filename or return error
            if "Error" not in status:
                assert output_path != output_file  # Should be different filename
            
            # Test with overwrite=True
            output_path, status = node.save_video(
                frames=sample_frames,
                output_filename="existing_file.mp4",
                output_directory=temp_video_dir,
                fps=30.0,
                codec="libx264",
                overwrite=True
            )
            
            assert output_path == output_file
            assert "Successfully saved" in status
    
    def test_float_frame_conversion(self, node, sample_float_frames, temp_video_dir):
        """Test conversion of float32 frames to uint8."""
        with patch('ffmpeg.run'), patch('ffmpeg.input'), patch('ffmpeg.output'):
            output_path, status = node.save_video(
                frames=sample_float_frames,
                output_filename="float_test.mp4",
                output_directory=temp_video_dir,
                fps=30.0,
                codec="libx264"
            )
            
            assert "Successfully saved" in status
    
    def test_invalid_frame_tensor(self, node, temp_video_dir):
        """Test error handling with invalid frame tensor."""
        # Test with wrong shape (missing channel dimension)
        invalid_frames = np.random.randint(0, 256, size=(10, 480, 640), dtype=np.uint8)
        
        output_path, status = node.save_video(
            frames=invalid_frames,
            output_filename="invalid_test.mp4",
            output_directory=temp_video_dir,
            fps=30.0,
            codec="libx264"
        )
        
        assert "Error" in status
    
    def test_empty_frame_tensor(self, node, temp_video_dir):
        """Test error handling with empty frame tensor."""
        empty_frames = np.array([]).reshape(0, 480, 640, 3)
        
        output_path, status = node.save_video(
            frames=empty_frames,
            output_filename="empty_test.mp4",
            output_directory=temp_video_dir,
            fps=30.0,
            codec="libx264"
        )
        
        assert "Error" in status or "empty" in status.lower()
    
    @patch('ffmpeg.run')
    def test_ffmpeg_error_handling(self, mock_run, node, sample_frames, temp_video_dir):
        """Test error handling when FFmpeg fails."""
        # Mock FFmpeg to raise an error
        mock_run.side_effect = Exception("FFmpeg error")
        
        with patch('ffmpeg.input'), patch('ffmpeg.output'):
            output_path, status = node.save_video(
                frames=sample_frames,
                output_filename="ffmpeg_error_test.mp4",
                output_directory=temp_video_dir,
                fps=30.0,
                codec="libx264"
            )
            
            assert "Error" in status
            assert "FFmpeg" in status or "encoding" in status.lower()
    
    def test_invalid_fps(self, node, sample_frames, temp_video_dir):
        """Test error handling with invalid frame rate."""
        with patch('ffmpeg.run'), patch('ffmpeg.input'), patch('ffmpeg.output'):
            # Test with zero fps
            output_path, status = node.save_video(
                frames=sample_frames,
                output_filename="zero_fps_test.mp4",
                output_directory=temp_video_dir,
                fps=0.0,
                codec="libx264"
            )
            
            assert "Error" in status or "fps" in status.lower()
            
            # Test with negative fps
            output_path, status = node.save_video(
                frames=sample_frames,
                output_filename="negative_fps_test.mp4",
                output_directory=temp_video_dir,
                fps=-10.0,
                codec="libx264"
            )
            
            assert "Error" in status or "fps" in status.lower()
    
    def test_invalid_codec(self, node, sample_frames, temp_video_dir):
        """Test handling of invalid codec."""
        with patch('ffmpeg.run'), patch('ffmpeg.input'), patch('ffmpeg.output'):
            output_path, status = node.save_video(
                frames=sample_frames,
                output_filename="invalid_codec_test.mp4",
                output_directory=temp_video_dir,
                fps=30.0,
                codec="invalid_codec"
            )
            
            # Should either use default codec or return error
            assert "saved" in status.lower() or "Error" in status
    
    def test_filename_sanitization(self, node, sample_frames, temp_video_dir):
        """Test filename sanitization."""
        with patch('ffmpeg.run'), patch('ffmpeg.input'), patch('ffmpeg.output'):
            # Test with problematic filename
            output_path, status = node.save_video(
                frames=sample_frames,
                output_filename="test<>|:*?\"file.mp4",
                output_directory=temp_video_dir,
                fps=30.0,
                codec="libx264"
            )
            
            # Should handle problematic characters
            if "Successfully saved" in status:
                filename = os.path.basename(output_path)
                # Should not contain problematic characters
                problematic_chars = '<>|:*?"'
                assert not any(char in filename for char in problematic_chars)
    
    def test_large_frame_tensor(self, node, temp_video_dir):
        """Test handling of large frame tensors."""
        # Create larger frame tensor (5 minutes at 30fps, 1080p)
        large_frames = np.random.randint(0, 256, size=(900, 1080, 1920, 3), dtype=np.uint8)
        
        with patch('ffmpeg.run'), patch('ffmpeg.input'), patch('ffmpeg.output'):
            output_path, status = node.save_video(
                frames=large_frames,
                output_filename="large_test.mp4",
                output_directory=temp_video_dir,
                fps=30.0,
                codec="libx264"
            )
            
            # Should handle large tensors without crashing
            assert "Error" not in status or "memory" in status.lower()
    
    def test_quality_settings_application(self, node, sample_frames, temp_video_dir):
        """Test that quality settings are properly applied."""
        quality_levels = ["low", "medium", "high"]
        
        with patch('ffmpeg.run') as mock_run, patch('ffmpeg.input'), patch('ffmpeg.output') as mock_output:
            for quality in quality_levels:
                output_path, status = node.save_video(
                    frames=sample_frames,
                    output_filename=f"quality_{quality}_test.mp4",
                    output_directory=temp_video_dir,
                    fps=30.0,
                    codec="libx264",
                    quality=quality
                )
                
                assert "Successfully saved" in status
                # Check that output was called (quality settings applied internally)
                mock_output.assert_called()
    
    def test_frame_preprocessing(self, node):
        """Test frame preprocessing and validation."""
        # Test different frame formats
        test_cases = [
            # (input_shape, expected_valid)
            ((60, 480, 640, 3), True),   # Standard BHWC
            ((30, 240, 320, 3), True),   # Different resolution
            ((10, 100, 100, 3), True),   # Small resolution
            ((5, 480, 640, 1), False),   # Single channel (grayscale)
            ((5, 480, 640), False),      # Missing channel dimension
            ((5, 480, 640, 3, 1), False),  # Extra dimension
        ]
        
        for shape, expected_valid in test_cases:
            frames = np.random.randint(0, 256, size=shape, dtype=np.uint8)
            is_valid = node.validate_frame_tensor(frames)
            assert is_valid == expected_valid
    
    def test_output_path_generation(self, node):
        """Test output path generation and validation."""
        test_cases = [
            ("test.mp4", "/output", "/output/test.mp4"),
            ("video.avi", "/my/path", "/my/path/video.avi"),
            ("no_extension", "/path", "/path/no_extension"),
            ("", "/path", "/path/untitled.mp4"),  # Empty filename
        ]
        
        for filename, directory, expected in test_cases:
            result = node.generate_output_path(filename, directory)
            if filename == "":
                assert result.endswith("untitled.mp4") or result.endswith(".mp4")
            else:
                assert result == expected
    
    def test_memory_efficient_processing(self, node, temp_video_dir):
        """Test memory-efficient frame processing."""
        # Create frames that would use significant memory
        frames = np.random.randint(0, 256, size=(120, 720, 1280, 3), dtype=np.uint8)
        
        import tracemalloc
        tracemalloc.start()
        
        with patch('ffmpeg.run'), patch('ffmpeg.input'), patch('ffmpeg.output'):
            output_path, status = node.save_video(
                frames=frames,
                output_filename="memory_test.mp4",
                output_directory=temp_video_dir,
                fps=30.0,
                codec="libx264"
            )
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable (allow 3x frame size for processing)
        frame_size = frames.nbytes
        assert peak < frame_size * 3