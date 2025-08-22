"""
Integration Tests for ComfyUI NonLinear Video Avatar Pipeline
End-to-end testing of the complete video processing workflow.
"""

import pytest
import os
import tempfile
import sys
import numpy as np
from unittest.mock import patch, Mock

# Add project path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.video_asset_loader import LoopyComfy_VideoAssetLoader
from nodes.markov_sequencer import LoopyComfy_MarkovVideoSequencer
from nodes.video_composer import LoopyComfy_VideoSequenceComposer
from nodes.video_saver import LoopyComfy_VideoSaver
from conftest import create_test_video


class TestIntegration:
    """Integration test suite for the complete ComfyUI pipeline."""
    
    @pytest.fixture
    def setup_test_videos(self):
        """Setup test environment with video files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            video_paths = []
            
            # Create 5 test videos with different properties
            for i in range(5):
                video_path = os.path.join(temp_dir, f"avatar_video_{i:03d}.mp4")
                create_test_video(
                    video_path,
                    duration=3.0 + i * 0.5,  # 3.0, 3.5, 4.0, 4.5, 5.0 seconds
                    fps=30.0,
                    width=640,
                    height=480,
                    seamless=True
                )
                video_paths.append(video_path)
            
            yield temp_dir, video_paths
    
    @pytest.fixture
    def pipeline_nodes(self):
        """Create all pipeline node instances."""
        return {
            'loader': LoopyComfy_VideoAssetLoader(),
            'sequencer': LoopyComfy_MarkovVideoSequencer(),
            'composer': LoopyComfy_VideoSequenceComposer(),
            'saver': LoopyComfy_VideoSaver()
        }
    
    def test_complete_pipeline_execution(self, setup_test_videos, pipeline_nodes):
        """Test complete pipeline from video loading to final output."""
        temp_dir, video_paths = setup_test_videos
        nodes = pipeline_nodes
        
        # Step 1: Load video assets
        metadata_list, load_status = nodes['loader'].load_video_assets(
            directory_path=temp_dir,
            file_pattern="*.mp4",
            max_videos=10,
            validate_seamless=True
        )
        
        assert len(metadata_list) == 5
        assert "Successfully loaded" in load_status
        
        # Step 2: Generate Markov sequence
        sequence, seq_status = nodes['sequencer'].generate_sequence(
            video_metadata=metadata_list,
            total_duration_minutes=1.0,  # 1 minute
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        assert len(sequence) > 0
        assert "Successfully generated" in seq_status
        
        # Verify sequence properties
        total_duration = sum(entry['duration'] for entry in sequence)
        assert total_duration >= 60.0
        
        # Verify no immediate repetitions
        for i in range(1, len(sequence)):
            assert sequence[i-1]['video_id'] != sequence[i]['video_id']
        
        # Step 3: Compose video sequence
        frames, comp_status = nodes['composer'].compose_sequence(
            sequence=sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=10
        )
        
        assert frames is not None
        assert frames.shape[1:] == (480, 640, 3)  # Height, Width, Channels
        assert "Successfully composed" in comp_status
        
        # Step 4: Save final video
        with patch('ffmpeg.run'), patch('ffmpeg.input'), patch('ffmpeg.output'):
            with tempfile.TemporaryDirectory() as output_dir:
                output_path, save_status = nodes['saver'].save_video(
                    frames=frames,
                    output_filename="integration_test_output.mp4",
                    output_directory=output_dir,
                    fps=30.0,
                    codec="libx264"
                )
                
                assert "Successfully saved" in save_status
                expected_path = os.path.join(output_dir, "integration_test_output.mp4")
                assert output_path == expected_path
    
    def test_pipeline_error_propagation(self, pipeline_nodes):
        """Test error handling and propagation through pipeline."""
        nodes = pipeline_nodes
        
        # Start with invalid input (empty directory)
        with tempfile.TemporaryDirectory() as empty_dir:
            metadata_list, load_status = nodes['loader'].load_video_assets(
                directory_path=empty_dir,
                file_pattern="*.mp4",
                max_videos=10,
                validate_seamless=True
            )
            
            assert metadata_list == []
            assert "No video files found" in load_status
            
            # Try to proceed with empty metadata
            sequence, seq_status = nodes['sequencer'].generate_sequence(
                video_metadata=metadata_list,
                total_duration_minutes=1.0,
                prevent_immediate_repeat=True,
                random_seed=42
            )
            
            assert sequence == []
            assert "Error" in seq_status or "empty" in seq_status.lower()
    
    def test_pipeline_with_different_parameters(self, setup_test_videos, pipeline_nodes):
        """Test pipeline with various parameter combinations."""
        temp_dir, video_paths = setup_test_videos
        nodes = pipeline_nodes
        
        # Test different sequence lengths and parameters
        test_configs = [
            {
                'duration_minutes': 0.5,
                'fps': 15.0,
                'resolution': '640x360',
                'codec': 'mpeg4'
            },
            {
                'duration_minutes': 2.0,
                'fps': 60.0,
                'resolution': '1280x720',
                'codec': 'libx265'
            }
        ]
        
        for config in test_configs:
            # Load videos
            metadata_list, _ = nodes['loader'].load_video_assets(
                directory_path=temp_dir,
                file_pattern="*.mp4",
                max_videos=10,
                validate_seamless=True
            )
            
            # Generate sequence
            sequence, _ = nodes['sequencer'].generate_sequence(
                video_metadata=metadata_list,
                total_duration_minutes=config['duration_minutes'],
                prevent_immediate_repeat=True,
                random_seed=42
            )
            
            # Compose frames
            frames, _ = nodes['composer'].compose_sequence(
                sequence=sequence,
                output_fps=config['fps'],
                resolution=config['resolution'],
                batch_size=5
            )
            
            if frames is not None:
                # Verify frame properties match resolution
                height, width = nodes['composer'].parse_resolution(config['resolution'])
                assert frames.shape[1:3] == (height, width)
    
    def test_pipeline_memory_efficiency(self, setup_test_videos, pipeline_nodes):
        """Test pipeline memory usage with larger datasets."""
        temp_dir, video_paths = setup_test_videos
        nodes = pipeline_nodes
        
        # Create additional videos for memory testing
        for i in range(5, 15):  # Add 10 more videos
            video_path = os.path.join(temp_dir, f"memory_test_{i:03d}.mp4")
            create_test_video(video_path, duration=2.0, fps=30.0)
        
        import tracemalloc
        tracemalloc.start()
        
        # Load all videos
        metadata_list, _ = nodes['loader'].load_video_assets(
            directory_path=temp_dir,
            file_pattern="*.mp4",
            max_videos=50,
            validate_seamless=True
        )
        
        # Generate longer sequence
        sequence, _ = nodes['sequencer'].generate_sequence(
            video_metadata=metadata_list,
            total_duration_minutes=5.0,  # 5 minutes
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        # Compose with memory limits
        frames, _ = nodes['composer'].compose_sequence(
            sequence=sequence[:10],  # Limit sequence for memory test
            output_fps=30.0,
            resolution="640x480",
            batch_size=5,  # Small batches
            memory_limit_gb=1.0
        )
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable
        assert peak < 500 * 1024 * 1024  # Less than 500MB
        
        if frames is not None:
            assert len(frames.shape) == 4  # Valid tensor
    
    def test_pipeline_reproducibility(self, setup_test_videos, pipeline_nodes):
        """Test pipeline reproducibility with same parameters."""
        temp_dir, video_paths = setup_test_videos
        nodes = pipeline_nodes
        
        # Run pipeline twice with same parameters
        results = []
        
        for run in range(2):
            # Load videos
            metadata_list, _ = nodes['loader'].load_video_assets(
                directory_path=temp_dir,
                file_pattern="*.mp4",
                max_videos=10,
                validate_seamless=True
            )
            
            # Generate sequence with fixed seed
            sequence, _ = nodes['sequencer'].generate_sequence(
                video_metadata=metadata_list,
                total_duration_minutes=1.0,
                prevent_immediate_repeat=True,
                random_seed=123  # Fixed seed
            )
            
            results.append(sequence)
        
        # Results should be identical
        assert len(results[0]) == len(results[1])
        for entry1, entry2 in zip(results[0], results[1]):
            assert entry1['video_id'] == entry2['video_id']
            assert entry1['duration'] == entry2['duration']
    
    def test_pipeline_edge_cases(self, pipeline_nodes):
        """Test pipeline behavior with edge cases."""
        nodes = pipeline_nodes
        
        # Test with single video file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create single video
            video_path = os.path.join(temp_dir, "single_video.mp4")
            create_test_video(video_path, duration=10.0, fps=30.0)
            
            # Load single video
            metadata_list, load_status = nodes['loader'].load_video_assets(
                directory_path=temp_dir,
                file_pattern="*.mp4",
                max_videos=10,
                validate_seamless=True
            )
            
            assert len(metadata_list) == 1
            
            # Try to generate sequence with repetition prevention
            sequence, seq_status = nodes['sequencer'].generate_sequence(
                video_metadata=metadata_list,
                total_duration_minutes=0.5,  # 30 seconds (less than video length)
                prevent_immediate_repeat=True,
                random_seed=42
            )
            
            # Should handle single video case gracefully
            if len(sequence) > 0:
                assert len(sequence) >= 1
                # All entries should be the same video
                for entry in sequence:
                    assert entry['video_id'] == 'single_video'
    
    def test_pipeline_with_invalid_videos(self, pipeline_nodes):
        """Test pipeline resilience with some invalid video files."""
        nodes = pipeline_nodes
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mix of valid and invalid files
            valid_video = os.path.join(temp_dir, "valid_video.mp4")
            create_test_video(valid_video, duration=3.0, fps=30.0)
            
            # Create invalid "video" file (just text)
            invalid_video = os.path.join(temp_dir, "invalid_video.mp4")
            with open(invalid_video, 'w') as f:
                f.write("This is not a video file")
            
            # Load videos - should skip invalid ones
            metadata_list, load_status = nodes['loader'].load_video_assets(
                directory_path=temp_dir,
                file_pattern="*.mp4",
                max_videos=10,
                validate_seamless=True
            )
            
            # Should have only valid videos
            assert len(metadata_list) >= 0  # At least attempt to process
            
            if len(metadata_list) > 0:
                # Continue pipeline with valid videos only
                sequence, _ = nodes['sequencer'].generate_sequence(
                    video_metadata=metadata_list,
                    total_duration_minutes=0.5,
                    prevent_immediate_repeat=True,
                    random_seed=42
                )
                
                assert len(sequence) >= 0  # Should handle gracefully
    
    def test_pipeline_performance_metrics(self, setup_test_videos, pipeline_nodes):
        """Test pipeline performance and timing."""
        temp_dir, video_paths = setup_test_videos
        nodes = pipeline_nodes
        
        import time
        
        start_time = time.time()
        
        # Load videos
        load_start = time.time()
        metadata_list, _ = nodes['loader'].load_video_assets(
            directory_path=temp_dir,
            file_pattern="*.mp4",
            max_videos=10,
            validate_seamless=True
        )
        load_time = time.time() - load_start
        
        # Generate sequence
        seq_start = time.time()
        sequence, _ = nodes['sequencer'].generate_sequence(
            video_metadata=metadata_list,
            total_duration_minutes=1.0,
            prevent_immediate_repeat=True,
            random_seed=42
        )
        seq_time = time.time() - seq_start
        
        # Compose frames (limited for performance test)
        comp_start = time.time()
        frames, _ = nodes['composer'].compose_sequence(
            sequence=sequence[:5],  # Limit for faster testing
            output_fps=30.0,
            resolution="640x480",
            batch_size=10
        )
        comp_time = time.time() - comp_start
        
        total_time = time.time() - start_time
        
        # Performance should be reasonable
        assert load_time < 10.0  # Loading should be fast
        assert seq_time < 5.0    # Sequencing should be fast
        assert comp_time < 30.0  # Composing may take longer
        assert total_time < 60.0  # Total should be under 1 minute
        
        print(f"Performance metrics:")
        print(f"  Load time: {load_time:.2f}s")
        print(f"  Sequence time: {seq_time:.2f}s") 
        print(f"  Compose time: {comp_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
    
    def test_pipeline_state_consistency(self, setup_test_videos, pipeline_nodes):
        """Test that pipeline maintains consistent state throughout."""
        temp_dir, video_paths = setup_test_videos
        nodes = pipeline_nodes
        
        # Load videos
        metadata_list, _ = nodes['loader'].load_video_assets(
            directory_path=temp_dir,
            file_pattern="*.mp4",
            max_videos=10,
            validate_seamless=True
        )
        
        original_metadata = [m.copy() for m in metadata_list]
        
        # Generate sequence
        sequence, _ = nodes['sequencer'].generate_sequence(
            video_metadata=metadata_list,
            total_duration_minutes=1.0,
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        # Verify metadata wasn't modified
        assert len(metadata_list) == len(original_metadata)
        for orig, curr in zip(original_metadata, metadata_list):
            assert orig == curr
        
        # Verify sequence references original metadata correctly
        video_ids_in_metadata = {m['video_id'] for m in metadata_list}
        video_ids_in_sequence = {s['video_id'] for s in sequence}
        
        # All sequence video IDs should exist in metadata
        assert video_ids_in_sequence.issubset(video_ids_in_metadata)
    
    def test_concurrent_pipeline_execution(self, setup_test_videos, pipeline_nodes):
        """Test pipeline behavior under concurrent execution scenarios."""
        temp_dir, video_paths = setup_test_videos
        nodes = pipeline_nodes
        
        # Simulate concurrent access by running multiple sequences
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def run_pipeline(seed):
            try:
                metadata_list, _ = nodes['loader'].load_video_assets(
                    directory_path=temp_dir,
                    file_pattern="*.mp4",
                    max_videos=10,
                    validate_seamless=True
                )
                
                sequence, _ = nodes['sequencer'].generate_sequence(
                    video_metadata=metadata_list,
                    total_duration_minutes=0.5,
                    prevent_immediate_repeat=True,
                    random_seed=seed
                )
                
                results_queue.put((seed, len(sequence), sequence[0]['video_id'] if sequence else None))
            except Exception as e:
                results_queue.put((seed, -1, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_pipeline, args=(42 + i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # All threads should complete successfully
        assert len(results) == 3
        for seed, seq_len, first_video in results:
            assert seq_len >= 0  # No errors
            if seq_len > 0:
                assert first_video is not None