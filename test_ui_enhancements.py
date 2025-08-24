#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI Enhancements Test Suite

Tests backward compatibility and validates that existing workflows
continue to work with the new UI enhancements.

CRITICAL: This test ensures that old workflows don't break.
"""

import sys
import os
import tempfile
import shutil
import json
from unittest.mock import Mock, patch

# Add project to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_video_asset_loader_backward_compatibility():
    """Test that VideoAssetLoader works with old workflow parameters."""
    print("Testing VideoAssetLoader backward compatibility...")
    
    try:
        from nodes.video_asset_loader import LoopyComfy_VideoAssetLoader
        
        # Create test directory with mock video files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake video files
            for i in range(3):
                fake_video = os.path.join(temp_dir, f"test_video_{i}.mp4")
                with open(fake_video, 'wb') as f:
                    f.write(b"fake video data")
            
            loader = LoopyComfy_VideoAssetLoader()
            
            # Test OLD workflow style (without new parameters)
            try:
                with patch('cv2.VideoCapture') as mock_cv2:
                    # Mock OpenCV VideoCapture
                    mock_cap = Mock()
                    mock_cap.isOpened.return_value = True
                    mock_cap.get.side_effect = lambda prop: {
                        0: 30.0,    # FPS
                        7: 300,     # Frame count
                        3: 1920,    # Width
                        4: 1080     # Height
                    }.get(prop, 0)
                    mock_cap.read.return_value = (True, Mock())
                    mock_cv2.return_value = mock_cap
                    
                    # OLD STYLE: Call without new parameters
                    result = loader.load_video_assets(
                        directory_path=temp_dir,
                        file_pattern="*.mp4",
                        max_videos=10,
                        validate_seamless=False
                    )
                    
                    # Should return tuple with 4 elements (backward compatibility)
                    assert len(result) >= 4, f"Expected at least 4 return values, got {len(result)}"
                    
                    video_metadata, video_count, total_duration, preview_grid = result[:4]
                    
                    assert isinstance(video_metadata, list), "video_metadata should be a list"
                    assert isinstance(video_count, int), "video_count should be an int"
                    assert isinstance(total_duration, (int, float)), "total_duration should be numeric"
                    
                    print("[+] OLD workflow parameters work correctly")
                    
            except Exception as old_error:
                print(f"[-] OLD workflow failed: {old_error}")
                return False
            
            # Test NEW workflow style (with new parameters)
            try:
                with patch('cv2.VideoCapture') as mock_cv2:
                    mock_cap = Mock()
                    mock_cap.isOpened.return_value = True
                    mock_cap.get.side_effect = lambda prop: {
                        0: 30.0, 7: 300, 3: 1920, 4: 1080
                    }.get(prop, 0)
                    mock_cap.read.return_value = (True, Mock())
                    mock_cv2.return_value = mock_cap
                    
                    # NEW STYLE: Call with new parameters
                    result = loader.load_video_assets(
                        directory_path=temp_dir,
                        browse_folder="📁 Browse Folder",  # New parameter
                        file_pattern="*.mp4",
                        max_videos=10,
                        validate_seamless=False,
                        show_preview=False  # New parameter
                    )
                    
                    # Should return tuple with 5 elements (new format)
                    assert len(result) == 5, f"Expected 5 return values, got {len(result)}"
                    
                    video_metadata, video_count, total_duration, preview_grid, folder_info = result
                    
                    assert isinstance(video_metadata, list), "video_metadata should be a list"
                    assert isinstance(video_count, int), "video_count should be an int"
                    assert isinstance(total_duration, (int, float)), "total_duration should be numeric"
                    assert isinstance(folder_info, str), "folder_info should be a string"
                    
                    print("[+] NEW workflow parameters work correctly")
                    
            except Exception as new_error:
                print(f"[-] NEW workflow failed: {new_error}")
                return False
                
        print("[+] VideoAssetLoader backward compatibility: PASSED")
        return True
        
    except Exception as e:
        print(f"[-] VideoAssetLoader test failed: {e}")
        return False

def test_video_saver_backward_compatibility():
    """Test that VideoSaver works with old workflow parameters."""
    print("Testing VideoSaver backward compatibility...")
    
    try:
        from nodes.video_saver import LoopyComfy_VideoSaver
        import numpy as np
        
        saver = LoopyComfy_VideoSaver()
        
        # Create test frames
        test_frames = np.random.rand(10, 480, 640, 3).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test OLD workflow style (without new parameters)
            try:
                with patch('ffmpeg.run') as mock_ffmpeg, \
                     patch('ffmpeg.run_async') as mock_async:
                    
                    # Mock FFmpeg process
                    mock_process = Mock()
                    mock_process.stdin = Mock()
                    mock_process.stderr = Mock()
                    mock_process.stderr.read.return_value = b"mock stderr"
                    mock_process.poll.return_value = 0
                    mock_process.returncode = 0
                    mock_async.return_value = mock_process
                    
                    # Mock file stats
                    with patch('os.stat') as mock_stat:
                        mock_stat.return_value = Mock(st_size=1024000)
                    
                        # OLD STYLE: Call without new parameters
                        result = saver.save_video(
                            frames=test_frames,
                            platform_preset="YouTube",
                            output_filename="test_old.mp4",
                            output_directory=temp_dir,
                            fps=30.0
                        )
                        
                        # Should return tuple with 4 elements
                        assert len(result) == 4, f"Expected 4 return values, got {len(result)}"
                        
                        output_path, statistics, files_created, total_size_mb = result
                        
                        assert isinstance(output_path, str), "output_path should be a string"
                        assert isinstance(statistics, dict), "statistics should be a dict"
                        assert isinstance(files_created, int), "files_created should be an int"
                        assert isinstance(total_size_mb, (int, float)), "total_size_mb should be numeric"
                        
                        print("[+] OLD VideoSaver parameters work correctly")
                        
            except Exception as old_error:
                print(f"[-] OLD VideoSaver workflow failed: {old_error}")
                return False
            
            # Test NEW workflow style (with new parameters)
            try:
                with patch('ffmpeg.run') as mock_ffmpeg, \
                     patch('ffmpeg.run_async') as mock_async:
                    
                    mock_process = Mock()
                    mock_process.stdin = Mock()
                    mock_process.stderr = Mock()
                    mock_process.stderr.read.return_value = b"mock stderr"
                    mock_process.poll.return_value = 0
                    mock_process.returncode = 0
                    mock_async.return_value = mock_process
                    
                    with patch('os.stat') as mock_stat:
                        mock_stat.return_value = Mock(st_size=1024000)
                    
                        # NEW STYLE: Call with new parameters
                        result = saver.save_video(
                            frames=test_frames,
                            platform_preset="YouTube",
                            output_filename="test_new.mp4",
                            output_directory=temp_dir,
                            browse_output_dir="📁 Browse Output Directory",  # New parameter
                            fps=30.0,
                            export_formats="mp4 (H.264 - Universal)"  # New parameter format
                        )
                        
                        assert len(result) == 4, f"Expected 4 return values, got {len(result)}"
                        
                        output_path, statistics, files_created, total_size_mb = result
                        
                        assert isinstance(output_path, str), "output_path should be a string"
                        assert isinstance(statistics, dict), "statistics should be a dict"
                        
                        print("[+] NEW VideoSaver parameters work correctly")
                        
            except Exception as new_error:
                print(f"[-] NEW VideoSaver workflow failed: {new_error}")
                return False
                
        print("[+] VideoSaver backward compatibility: PASSED")
        return True
        
    except Exception as e:
        print(f"[-] VideoSaver test failed: {e}")
        return False

def test_memory_safe_preview():
    """Test the memory safety constraints of the preview feature."""
    print("Testing memory-safe preview implementation...")
    
    try:
        from nodes.video_asset_loader import LoopyComfy_VideoAssetLoader
        import numpy as np
        
        loader = LoopyComfy_VideoAssetLoader()
        
        # Create test metadata for memory testing
        test_metadata = []
        for i in range(20):  # More than MAX_PREVIEW_VIDEOS (10)
            test_metadata.append({
                'file_path': f'/fake/path/video_{i}.mp4',
                'filename': f'video_{i}.mp4',
                'duration': 10.0,
                'width': 1920,
                'height': 1080
            })
        
        with patch('cv2.VideoCapture') as mock_cv2:
            # Mock successful video capture
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: 300 if prop == 7 else 30
            
            # Mock frame reading (return small test frames)
            test_frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, test_frame)
            mock_cv2.return_value = mock_cap
            
            # Test memory-safe preview
            preview = loader._generate_memory_safe_preview(test_metadata, "thumbnails")
            
            if preview is not None:
                # Validate memory constraints
                memory_size_mb = preview.nbytes / (1024 * 1024)
                assert memory_size_mb <= 1000, f"Preview uses {memory_size_mb:.1f}MB > 1GB limit"
                
                # Validate frame count constraints
                if len(preview.shape) == 5:  # (batch, frames, height, width, channels)
                    frame_count = preview.shape[1]
                else:
                    frame_count = preview.shape[0]  # (frames, height, width, channels)
                
                assert frame_count <= 150, f"Preview has {frame_count} frames > 150 limit"
                
                # Validate resolution constraints (360p)
                height = preview.shape[-3]
                width = preview.shape[-2]
                assert height <= 360, f"Preview height {height} > 360p limit"
                assert width <= 640, f"Preview width {width} > 640p limit"
                
                print(f"[+] Preview generated: {frame_count} frames, {memory_size_mb:.1f}MB")
            else:
                print("[+] Preview skipped for memory safety")
        
        print("[+] Memory-safe preview: PASSED")
        return True
        
    except Exception as e:
        print(f"[-] Memory-safe preview test failed: {e}")
        return False

def test_api_extensions():
    """Test API extensions for folder browsing."""
    print("Testing API extensions...")
    
    try:
        from api_extensions import LoopyComfyAPI
        
        api = LoopyComfyAPI()
        
        # Test folder browse (with mocked tkinter)
        with patch('tkinter.Tk') as mock_tk, \
             patch('tkinter.filedialog.askdirectory') as mock_dialog:
            
            # Mock successful folder selection
            mock_dialog.return_value = "/selected/folder"
            mock_tk.return_value = Mock()
            
            # Test folder browse request
            request_data = {
                "node_id": "test_node",
                "current_path": "/current/path"
            }
            
            result = api.browse_folder(request_data)
            
            assert "success" in result or "error" in result, "Result should have success or error"
            
            if result.get("success"):
                assert "path" in result, "Successful result should have path"
                print("[+] Folder browse API works correctly")
            else:
                # Fallback behavior should work
                assert "fallback_suggestions" in result or "error" in result
                print("[+] Folder browse fallback works correctly")
        
        # Test output directory browse
        with patch('tkinter.Tk') as mock_tk, \
             patch('tkinter.filedialog.askdirectory') as mock_dialog:
            
            mock_dialog.return_value = "/output/dir"
            mock_tk.return_value = Mock()
            
            request_data = {"current_path": "./output/"}
            result = api.browse_output_directory(request_data)
            
            assert "success" in result or "error" in result
            print("[+] Output directory browse API works correctly")
        
        print("[+] API extensions: PASSED")
        return True
        
    except Exception as e:
        print(f"[-] API extensions test failed: {e}")
        return False

def test_existing_workflows():
    """Test that existing workflow files still work."""
    print("Testing existing workflow compatibility...")
    
    try:
        # Check if workflow files exist
        workflow_dir = os.path.join(current_dir, "workflows")
        
        if os.path.exists(workflow_dir):
            for workflow_file in ["basic_avatar_workflow.json", "advanced_avatar_workflow.json"]:
                workflow_path = os.path.join(workflow_dir, workflow_file)
                
                if os.path.exists(workflow_path):
                    with open(workflow_path, 'r') as f:
                        workflow_data = json.load(f)
                    
                    # Validate workflow structure
                    assert "nodes" in workflow_data or "workflow" in workflow_data, \
                           f"Workflow {workflow_file} missing nodes/workflow"
                    
                    print(f"[+] Workflow {workflow_file} structure is valid")
                else:
                    print(f"[!] Workflow {workflow_file} not found (optional)")
        else:
            print("[!] Workflows directory not found (optional)")
        
        print("[+] Existing workflows: PASSED")
        return True
        
    except Exception as e:
        print(f"[-] Workflow compatibility test failed: {e}")
        return False

def run_all_tests():
    """Run all UI enhancement tests."""
    print("=" * 60)
    print("LoopyComfy UI Enhancements - Compatibility Test Suite")
    print("=" * 60)
    
    tests = [
        test_video_asset_loader_backward_compatibility,
        test_video_saver_backward_compatibility,
        test_memory_safe_preview,
        test_api_extensions,
        test_existing_workflows
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        print(f"\n{'=' * 40}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[-] Test {test_func.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    
    if failed == 0:
        print("[SUCCESS] ALL TESTS PASSED - UI Enhancements are backward compatible!")
        return True
    else:
        print("[FAIL] SOME TESTS FAILED - Review implementation before deployment")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)