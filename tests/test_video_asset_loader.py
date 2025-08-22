"""
Tests for VideoAssetLoader Node - ComfyUI NonLinear Video Avatar
"""

import pytest
import os
import tempfile
import cv2
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.video_asset_loader import LoopyComfy_VideoAssetLoader
from conftest import create_test_video


class TestVideoAssetLoader:
    """Test suite for VideoAssetLoader ComfyUI node."""
    
    @pytest.fixture
    def node(self):
        """Create VideoAssetLoader node instance."""
        return LoopyComfy_VideoAssetLoader()
    
    @pytest.fixture
    def test_videos_dir(self, temp_video_dir):
        """Create test videos in temporary directory."""
        video_files = []
        
        # Create test videos with varying properties
        for i in range(3):
            video_path = os.path.join(temp_video_dir, f"test_video_{i:03d}.mp4")
            create_test_video(
                video_path, 
                duration=3.0 + i,  # 3, 4, 5 seconds
                fps=30.0,
                width=640,
                height=480,
                seamless=True
            )
            video_files.append(video_path)
        
        # Create a non-seamless video
        non_seamless_path = os.path.join(temp_video_dir, "non_seamless.mp4")
        create_test_video(
            non_seamless_path,
            duration=2.0,
            fps=30.0,
            width=640,
            height=480,
            seamless=False
        )
        video_files.append(non_seamless_path)
        
        return temp_video_dir, video_files
    
    def test_input_types_structure(self, node):
        """Test INPUT_TYPES returns correct structure."""
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        required = input_types["required"]
        
        # Check required fields
        assert "directory_path" in required
        assert "file_pattern" in required
        assert "max_videos" in required
        assert "validate_seamless" in required
        
        # Check field types and defaults
        assert required["directory_path"][0] == "STRING"
        assert required["file_pattern"][1]["default"] == "*.mp4"
        assert required["max_videos"][0] == "INT"
        assert required["validate_seamless"][0] == "BOOLEAN"
    
    def test_return_types(self, node):
        """Test RETURN_TYPES returns correct types."""
        return_types = node.RETURN_TYPES
        
        assert return_types == ("VIDEO_METADATA_LIST", "STRING")
    
    def test_return_names(self, node):
        """Test RETURN_NAMES returns correct names."""
        return_names = node.RETURN_NAMES
        
        assert return_names == ("video_metadata", "status_message")
    
    def test_function_name(self, node):
        """Test FUNCTION returns correct function name."""
        function_name = node.FUNCTION
        
        assert function_name == "load_video_assets"
    
    def test_category(self, node):
        """Test CATEGORY returns correct category."""
        category = node.CATEGORY
        
        assert category == "video/avatar"
    
    def test_load_video_assets_basic(self, node, test_videos_dir):
        """Test basic video asset loading."""
        temp_dir, video_files = test_videos_dir
        
        # Load assets
        metadata_list, status = node.load_video_assets(
            directory_path=temp_dir,
            file_pattern="*.mp4",
            max_videos=10,
            validate_seamless=True
        )
        
        assert isinstance(metadata_list, list)
        assert len(metadata_list) == 4  # 3 seamless + 1 non-seamless
        assert "Successfully loaded" in status
        
        # Check metadata structure
        for metadata in metadata_list[:3]:  # First 3 are seamless
            assert "video_id" in metadata
            assert "file_path" in metadata
            assert "filename" in metadata
            assert "duration" in metadata
            assert "fps" in metadata
            assert "frame_count" in metadata
            assert "width" in metadata
            assert "height" in metadata
            assert "resolution" in metadata
            assert "file_size" in metadata
            assert "is_seamless" in metadata
            assert metadata["is_seamless"] == True
    
    def test_load_video_assets_with_pattern(self, node, test_videos_dir):
        """Test loading with specific file pattern."""
        temp_dir, video_files = test_videos_dir
        
        # Load only files matching specific pattern
        metadata_list, status = node.load_video_assets(
            directory_path=temp_dir,
            file_pattern="test_video_*.mp4",
            max_videos=10,
            validate_seamless=True
        )
        
        assert len(metadata_list) == 3  # Only test_video_*.mp4 files
        
        for metadata in metadata_list:
            assert metadata["filename"].startswith("test_video_")
    
    def test_load_video_assets_max_limit(self, node, test_videos_dir):
        """Test max_videos limit."""
        temp_dir, video_files = test_videos_dir
        
        # Limit to 2 videos
        metadata_list, status = node.load_video_assets(
            directory_path=temp_dir,
            file_pattern="*.mp4",
            max_videos=2,
            validate_seamless=True
        )
        
        assert len(metadata_list) == 2
        assert "Limited to 2" in status
    
    def test_load_video_assets_skip_validation(self, node, test_videos_dir):
        """Test loading without seamless validation."""
        temp_dir, video_files = test_videos_dir
        
        metadata_list, status = node.load_video_assets(
            directory_path=temp_dir,
            file_pattern="*.mp4",
            max_videos=10,
            validate_seamless=False
        )
        
        # All videos should be included (including non-seamless)
        assert len(metadata_list) == 4
        
        # All should be marked as seamless=True when validation is skipped
        for metadata in metadata_list:
            assert metadata["is_seamless"] == True
    
    def test_empty_directory(self, node):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_list, status = node.load_video_assets(
                directory_path=temp_dir,
                file_pattern="*.mp4",
                max_videos=10,
                validate_seamless=True
            )
            
            assert metadata_list == []
            assert "No video files found" in status
    
    def test_invalid_directory(self, node):
        """Test loading from non-existent directory."""
        metadata_list, status = node.load_video_assets(
            directory_path="/non/existent/path",
            file_pattern="*.mp4",
            max_videos=10,
            validate_seamless=True
        )
        
        assert metadata_list == []
        assert "Error" in status
    
    def test_seamless_validation_logic(self, node):
        """Test seamless loop validation logic."""
        # Test the is_seamless_loop function directly
        # Create a simple test video
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "test.mp4")
            create_test_video(video_path, duration=2.0, seamless=True)
            
            is_seamless = node.is_seamless_loop(video_path)
            assert isinstance(is_seamless, bool)
    
    @patch('cv2.VideoCapture')
    def test_extract_metadata_error_handling(self, mock_video_capture, node):
        """Test error handling in metadata extraction."""
        # Mock video capture to fail
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        metadata = node.extract_metadata("/fake/path/test.mp4")
        
        assert metadata is None
    
    @patch('cv2.VideoCapture')  
    def test_extract_metadata_success(self, mock_video_capture, node):
        """Test successful metadata extraction."""
        # Mock successful video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 150.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0
        }.get(prop, 0.0)
        mock_video_capture.return_value = mock_cap
        
        # Mock os.path.getsize
        with patch('os.path.getsize', return_value=1024000):
            metadata = node.extract_metadata("/fake/path/test_video.mp4")
            
            assert metadata is not None
            assert metadata["video_id"] == "test_video"
            assert metadata["duration"] == 5.0  # 150 frames / 30 fps
            assert metadata["fps"] == 30.0
            assert metadata["frame_count"] == 150
            assert metadata["width"] == 1920
            assert metadata["height"] == 1080
            assert metadata["resolution"] == "1920x1080"
            assert metadata["file_size"] == 1024000
    
    def test_seamless_loop_detection_failure(self, node):
        """Test seamless loop detection with non-existent file."""
        is_seamless = node.is_seamless_loop("/non/existent/file.mp4")
        assert is_seamless == False
    
    def test_large_file_handling(self, node, test_videos_dir):
        """Test handling of multiple video files."""
        temp_dir, video_files = test_videos_dir
        
        # Create additional video files
        for i in range(5, 8):
            video_path = os.path.join(temp_dir, f"extra_video_{i:03d}.mp4")
            create_test_video(video_path, duration=1.0, seamless=True)
        
        metadata_list, status = node.load_video_assets(
            directory_path=temp_dir,
            file_pattern="*.mp4",
            max_videos=5,  # Limit to 5
            validate_seamless=True
        )
        
        assert len(metadata_list) == 5
        assert "Limited to 5" in status
    
    def test_video_id_generation(self, node):
        """Test video ID generation from filenames."""
        test_cases = [
            ("test_video_001.mp4", "test_video_001"),
            ("avatar.mov", "avatar"), 
            ("complex.name.with.dots.avi", "complex.name.with.dots"),
            ("no_extension", "no_extension")
        ]
        
        for filename, expected_id in test_cases:
            video_id = node.generate_video_id(filename)
            assert video_id == expected_id
    
    def test_supported_formats(self, node):
        """Test that various video formats are handled."""
        test_formats = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
        
        for fmt in test_formats:
            filename = f"test{fmt}"
            video_id = node.generate_video_id(filename)
            assert video_id == "test"
    
    def test_metadata_consistency(self, node, test_videos_dir):
        """Test that metadata is consistent across multiple loads."""
        temp_dir, video_files = test_videos_dir
        
        # Load same directory twice
        metadata1, _ = node.load_video_assets(temp_dir, "*.mp4", 10, True)
        metadata2, _ = node.load_video_assets(temp_dir, "*.mp4", 10, True)
        
        assert len(metadata1) == len(metadata2)
        
        # Sort by video_id for comparison
        metadata1.sort(key=lambda x: x["video_id"])
        metadata2.sort(key=lambda x: x["video_id"])
        
        for m1, m2 in zip(metadata1, metadata2):
            assert m1["video_id"] == m2["video_id"]
            assert m1["duration"] == m2["duration"]
            assert m1["fps"] == m2["fps"]
            assert m1["frame_count"] == m2["frame_count"]
    
    def test_memory_efficiency(self, node, test_videos_dir):
        """Test that loading doesn't consume excessive memory."""
        temp_dir, video_files = test_videos_dir
        
        # Create more video files to test memory usage
        for i in range(10, 20):
            video_path = os.path.join(temp_dir, f"memory_test_{i:03d}.mp4") 
            create_test_video(video_path, duration=1.0, seamless=True)
        
        import tracemalloc
        tracemalloc.start()
        
        metadata_list, status = node.load_video_assets(
            directory_path=temp_dir,
            file_pattern="*.mp4",
            max_videos=50,
            validate_seamless=True
        )
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Check that we successfully loaded videos
        assert len(metadata_list) > 10
        
        # Memory usage should be reasonable (less than 50MB for metadata)
        assert peak < 50 * 1024 * 1024  # 50MB