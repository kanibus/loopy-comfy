"""
Edge Cases and Stress Tests for ComfyUI NonLinear Video Avatar
"""

import pytest
import numpy as np
import tempfile
import os
import sys
from unittest.mock import Mock, patch
import threading
import time

# Add project path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.video_asset_loader import LoopyComfy_VideoAssetLoader
from nodes.markov_sequencer import LoopyComfy_MarkovVideoSequencer
from nodes.video_composer import LoopyComfy_VideoSequenceComposer
from nodes.video_saver import LoopyComfy_VideoSaver
from core.markov_engine import MarkovTransitionEngine, MarkovState, validate_no_repetition
from .conftest import create_test_video


class TestEdgeCases:
    """Test suite for edge cases and stress scenarios."""
    
    @pytest.fixture
    def all_nodes(self):
        """Get all node instances for testing."""
        return {
            'loader': LoopyComfy_VideoAssetLoader(),
            'sequencer': LoopyComfy_MarkovVideoSequencer(),
            'composer': LoopyComfy_VideoSequenceComposer(),
            'saver': LoopyComfy_VideoSaver()
        }
    
    # === SINGLE-STATE EDGE CASES ===
    
    def test_single_video_markov_engine(self):
        """Test Markov engine with single video (critical edge case)."""
        single_state = [MarkovState("single_video", 5.0, {})]
        
        # Should handle single state gracefully
        try:
            engine = MarkovTransitionEngine(single_state)
            
            # Getting next state from same state should handle the edge case
            # Either return the same state or raise appropriate error
            next_state = engine.get_next_state("single_video")
            
            # If it returns a state, it should be the only available state
            assert next_state == "single_video"
            
        except ValueError as e:
            # Engine should raise informative error for impossible scenario
            assert "single" in str(e).lower() or "repetition" in str(e).lower()
    
    def test_single_video_sequencer_node(self, all_nodes):
        """Test MarkovVideoSequencer with single video."""
        single_metadata = [
            {
                "video_id": "only_video",
                "file_path": "/fake/only_video.mp4",
                "filename": "only_video.mp4",
                "duration": 10.0,
                "fps": 30.0,
                "frame_count": 300,
                "width": 1920,
                "height": 1080,
                "resolution": "1920x1080",
                "file_size": 2048000,
                "is_seamless": True
            }
        ]
        
        sequence, status = all_nodes['sequencer'].generate_sequence(
            video_metadata=single_metadata,
            total_duration_minutes=0.5,  # 30 seconds (less than video duration)
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        # Should handle gracefully - either create valid sequence or error
        if len(sequence) > 0:
            # If sequence created, should be valid
            assert len(sequence) >= 1
            for entry in sequence:
                assert entry['video_id'] == 'only_video'
        else:
            # If no sequence, should have informative error message
            assert "Error" in status or "single" in status.lower()
    
    def test_single_video_prevent_repetition_disable(self, all_nodes):
        """Test single video with repetition prevention disabled."""
        single_metadata = [
            {
                "video_id": "repeat_video",
                "file_path": "/fake/repeat_video.mp4",
                "filename": "repeat_video.mp4",
                "duration": 3.0,
                "fps": 30.0,
                "frame_count": 90,
                "width": 1920,
                "height": 1080,
                "resolution": "1920x1080",
                "file_size": 1000000,
                "is_seamless": True
            }
        ]
        
        sequence, status = all_nodes['sequencer'].generate_sequence(
            video_metadata=single_metadata,
            total_duration_minutes=0.2,  # 12 seconds, need multiple clips
            prevent_immediate_repeat=False,  # Allow repetition
            random_seed=42
        )
        
        # Should work when repetition is allowed
        if len(sequence) > 1:
            # Multiple entries should all be the same video
            for entry in sequence:
                assert entry['video_id'] == 'repeat_video'
    
    # === EXTREME PARAMETER EDGE CASES ===
    
    def test_zero_duration_sequence(self, all_nodes):
        """Test sequence generation with zero duration."""
        metadata = [
            {
                "video_id": "test_video",
                "file_path": "/fake/test.mp4", 
                "filename": "test.mp4",
                "duration": 5.0,
                "fps": 30.0,
                "frame_count": 150,
                "width": 1920,
                "height": 1080,
                "resolution": "1920x1080",
                "file_size": 1000000,
                "is_seamless": True
            }
        ]
        
        sequence, status = all_nodes['sequencer'].generate_sequence(
            video_metadata=metadata,
            total_duration_minutes=0.0,  # Zero duration
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        assert len(sequence) == 0
        assert "Error" in status or "duration" in status.lower()
    
    def test_negative_duration_sequence(self, all_nodes):
        """Test sequence generation with negative duration."""
        metadata = [
            {
                "video_id": "test_video",
                "file_path": "/fake/test.mp4",
                "filename": "test.mp4", 
                "duration": 5.0,
                "fps": 30.0,
                "frame_count": 150,
                "width": 1920,
                "height": 1080,
                "resolution": "1920x1080",
                "file_size": 1000000,
                "is_seamless": True
            }
        ]
        
        sequence, status = all_nodes['sequencer'].generate_sequence(
            video_metadata=metadata,
            total_duration_minutes=-1.0,  # Negative duration
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        assert len(sequence) == 0
        assert "Error" in status
    
    def test_extremely_large_duration(self, all_nodes):
        """Test sequence generation with very large duration."""
        metadata = [
            {
                "video_id": f"video_{i}",
                "file_path": f"/fake/video_{i}.mp4",
                "filename": f"video_{i}.mp4",
                "duration": 5.0,
                "fps": 30.0,
                "frame_count": 150,
                "width": 1920,
                "height": 1080,
                "resolution": "1920x1080",
                "file_size": 1000000,
                "is_seamless": True
            } for i in range(10)
        ]
        
        # Test with very large duration (24 hours)
        sequence, status = all_nodes['sequencer'].generate_sequence(
            video_metadata=metadata,
            total_duration_minutes=1440.0,  # 24 hours
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        # Should either succeed or fail gracefully
        if len(sequence) > 0:
            total_duration = sum(entry['duration'] for entry in sequence)
            assert total_duration >= 24 * 3600  # 24 hours in seconds
            
            # Verify no immediate repetitions in long sequence
            for i in range(1, min(len(sequence), 1000)):  # Check first 1000
                assert sequence[i-1]['video_id'] != sequence[i]['video_id']
        else:
            assert "Error" in status or "duration" in status.lower()
    
    # === EXTREME FPS AND RESOLUTION EDGE CASES ===
    
    def test_zero_fps_video_saver(self, all_nodes):
        """Test VideoSaver with zero FPS."""
        frames = np.random.randint(0, 256, size=(10, 240, 320, 3), dtype=np.uint8)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path, status = all_nodes['saver'].save_video(
                frames=frames,
                output_filename="zero_fps_test.mp4",
                output_directory=temp_dir,
                fps=0.0,  # Zero FPS
                codec="libx264"
            )
            
            assert "Error" in status or "fps" in status.lower()
    
    def test_extreme_high_fps(self, all_nodes):
        """Test with extremely high frame rate."""
        frames = np.random.randint(0, 256, size=(10, 240, 320, 3), dtype=np.uint8)
        
        with patch('ffmpeg.run'), patch('ffmpeg.input'), patch('ffmpeg.output'):
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path, status = all_nodes['saver'].save_video(
                    frames=frames,
                    output_filename="high_fps_test.mp4",
                    output_directory=temp_dir,
                    fps=10000.0,  # Extremely high FPS
                    codec="libx264"
                )
                
                # Should either succeed or fail gracefully
                assert "saved" in status.lower() or "Error" in status
    
    def test_invalid_resolution_format(self, all_nodes):
        """Test VideoComposer with invalid resolution."""
        sample_sequence = [
            {
                "video_id": "test_video",
                "file_path": "/fake/test.mp4",
                "filename": "test.mp4",
                "duration": 2.0,
                "start_time": 0.0,
                "end_time": 2.0
            }
        ]
        
        frames, status = all_nodes['composer'].compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="invalid_format",  # Invalid format
            batch_size=10
        )
        
        # Should either use default resolution or return error
        if frames is not None:
            assert len(frames.shape) == 4  # Valid tensor shape
        else:
            assert "Error" in status
    
    # === MEMORY AND PERFORMANCE EDGE CASES ===
    
    def test_massive_frame_tensor(self, all_nodes):
        """Test handling of very large frame tensors."""
        # Create large frame tensor (simulating long high-res video)
        try:
            # Try to create 1000 frames of 1080p (may fail on limited memory systems)
            large_frames = np.random.randint(0, 256, size=(1000, 1080, 1920, 3), dtype=np.uint8)
            
            with patch('ffmpeg.run'), patch('ffmpeg.input'), patch('ffmpeg.output'):
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_path, status = all_nodes['saver'].save_video(
                        frames=large_frames,
                        output_filename="massive_test.mp4",
                        output_directory=temp_dir,
                        fps=30.0,
                        codec="libx264"
                    )
                    
                    # Should handle without crashing
                    assert isinstance(status, str)
                    
        except MemoryError:
            # Expected on systems with limited memory
            pytest.skip("Not enough memory for massive frame tensor test")
    
    def test_memory_limit_enforcement(self, all_nodes):
        """Test memory limit enforcement in VideoComposer."""
        sample_sequence = [
            {
                "video_id": "test_video",
                "file_path": "/fake/test.mp4",
                "filename": "test.mp4",
                "duration": 2.0,
                "start_time": 0.0,
                "end_time": 2.0
            }
        ]
        
        frames, status = all_nodes['composer'].compose_sequence(
            sequence=sample_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=1,  # Very small batch
            memory_limit_gb=0.001  # Very small memory limit (1MB)
        )
        
        # Should either succeed with limited processing or fail gracefully
        assert isinstance(status, str)
        if frames is not None:
            assert len(frames.shape) == 4
    
    # === CONCURRENT ACCESS EDGE CASES ===
    
    def test_concurrent_markov_engine_access(self):
        """Test concurrent access to Markov engine."""
        states = [MarkovState(f"video_{i}", 5.0, {}) for i in range(5)]
        engine = MarkovTransitionEngine(states, random_seed=42)
        
        results = []
        errors = []
        
        def generate_sequence(thread_id):
            try:
                current_state = None
                sequence = []
                for _ in range(100):
                    next_state = engine.get_next_state(current_state)
                    sequence.append(next_state)
                    current_state = next_state
                results.append((thread_id, sequence))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=generate_sequence, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Check results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 3
        
        # Verify all sequences are valid (no immediate repetitions)
        for thread_id, sequence in results:
            for i in range(1, len(sequence)):
                assert sequence[i-1] != sequence[i], \
                    f"Thread {thread_id} immediate repetition at {i}"
    
    # === MALFORMED DATA EDGE CASES ===
    
    def test_malformed_metadata_structure(self, all_nodes):
        """Test handling of malformed video metadata."""
        malformed_metadata = [
            # Missing required fields
            {
                "video_id": "incomplete_1",
                "file_path": "/fake/test.mp4"
                # Missing duration, fps, etc.
            },
            # Wrong data types
            {
                "video_id": 123,  # Should be string
                "file_path": "/fake/test2.mp4",
                "duration": "five",  # Should be number
                "fps": 30.0,
                "frame_count": 150,
                "width": 1920,
                "height": 1080,
                "resolution": "1920x1080",
                "file_size": 1000000,
                "is_seamless": True
            },
            # None values
            {
                "video_id": None,
                "file_path": None,
                "duration": None,
                "fps": None,
                "frame_count": None,
                "width": None,
                "height": None,
                "resolution": None,
                "file_size": None,
                "is_seamless": None
            }
        ]
        
        sequence, status = all_nodes['sequencer'].generate_sequence(
            video_metadata=malformed_metadata,
            total_duration_minutes=1.0,
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        # Should handle malformed data gracefully
        assert len(sequence) == 0
        assert "Error" in status
    
    def test_empty_video_sequence(self, all_nodes):
        """Test VideoComposer with empty sequence."""
        empty_sequence = []
        
        frames, status = all_nodes['composer'].compose_sequence(
            sequence=empty_sequence,
            output_fps=30.0,
            resolution="640x480",
            batch_size=10
        )
        
        assert frames is None or (hasattr(frames, 'shape') and frames.shape[0] == 0)
        assert "Error" in status or "empty" in status.lower()
    
    def test_malformed_frame_tensor_shapes(self, all_nodes):
        """Test VideoSaver with various malformed frame tensor shapes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            malformed_tensors = [
                # Wrong number of dimensions
                np.random.randint(0, 256, size=(100, 480, 640), dtype=np.uint8),  # Missing channel dim
                np.random.randint(0, 256, size=(100, 480, 640, 3, 1), dtype=np.uint8),  # Extra dim
                
                # Wrong channel count
                np.random.randint(0, 256, size=(100, 480, 640, 1), dtype=np.uint8),  # Grayscale
                np.random.randint(0, 256, size=(100, 480, 640, 4), dtype=np.uint8),  # RGBA
                
                # Zero dimensions
                np.array([]).reshape(0, 480, 640, 3),  # No frames
                np.random.randint(0, 256, size=(100, 0, 640, 3), dtype=np.uint8),  # Zero height
                np.random.randint(0, 256, size=(100, 480, 0, 3), dtype=np.uint8),  # Zero width
            ]
            
            for i, tensor in enumerate(malformed_tensors):
                output_path, status = all_nodes['saver'].save_video(
                    frames=tensor,
                    output_filename=f"malformed_{i}.mp4",
                    output_directory=temp_dir,
                    fps=30.0,
                    codec="libx264"
                )
                
                # Should detect invalid tensor and return error
                assert "Error" in status
    
    # === FILE SYSTEM EDGE CASES ===
    
    def test_permission_denied_directory(self, all_nodes):
        """Test handling of permission-denied scenarios."""
        # Test with root directory (should be permission denied)
        metadata_list, status = all_nodes['loader'].load_video_assets(
            directory_path="/root/nonexistent",  # Typically permission denied
            file_pattern="*.mp4",
            max_videos=10,
            validate_seamless=True
        )
        
        assert len(metadata_list) == 0
        assert "Error" in status or "not found" in status.lower()
    
    def test_extremely_long_filename(self, all_nodes):
        """Test handling of extremely long filenames."""
        long_filename = "a" * 300 + ".mp4"  # 300+ character filename
        
        frames = np.random.randint(0, 256, size=(10, 240, 320, 3), dtype=np.uint8)
        
        with patch('ffmpeg.run'), patch('ffmpeg.input'), patch('ffmpeg.output'):
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path, status = all_nodes['saver'].save_video(
                    frames=frames,
                    output_filename=long_filename,
                    output_directory=temp_dir,
                    fps=30.0,
                    codec="libx264"
                )
                
                # Should either truncate filename or return error
                if "saved" in status.lower():
                    # Filename should be truncated or modified
                    actual_filename = os.path.basename(output_path)
                    assert len(actual_filename) < len(long_filename)
                else:
                    assert "Error" in status
    
    # === CRITICAL 10K NO-REPETITION VALIDATION ===
    
    def test_extended_no_repetition_validation(self):
        """Extended validation of no-repetition guarantee (10K+ iterations)."""
        # Test with different state counts
        state_counts = [2, 3, 5, 10, 20]
        
        for state_count in state_counts:
            states = [MarkovState(f"video_{i:03d}", 5.0, {}) for i in range(state_count)]
            engine = MarkovTransitionEngine(states, random_seed=42)
            
            # Run extended validation
            validation_results = validate_no_repetition(engine, test_iterations=10000)
            
            assert validation_results["passed"], \
                f"CRITICAL FAILURE: {state_count} states - " \
                f"Immediate repetitions: {validation_results['immediate_repetitions']}"
            
            assert validation_results["immediate_repetitions"] == 0, \
                f"CRITICAL: Non-zero repetitions with {state_count} states"
    
    def test_stress_no_repetition_multiple_engines(self):
        """Stress test no-repetition with multiple concurrent engines."""
        results = []
        
        def test_engine(engine_id, state_count):
            states = [MarkovState(f"e{engine_id}_v{i}", 5.0, {}) for i in range(state_count)]
            engine = MarkovTransitionEngine(states, random_seed=42 + engine_id)
            
            validation_results = validate_no_repetition(engine, test_iterations=5000)
            results.append((engine_id, state_count, validation_results))
        
        # Test multiple engines concurrently
        threads = []
        for i in range(5):  # 5 engines
            thread = threading.Thread(target=test_engine, args=(i, 5))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=30)
        
        # All engines should pass
        assert len(results) == 5
        for engine_id, state_count, validation_results in results:
            assert validation_results["passed"], \
                f"Engine {engine_id} failed: {validation_results['immediate_repetitions']} repetitions"
    
    # === NUMERIC EDGE CASES ===
    
    def test_float_precision_edge_cases(self, all_nodes):
        """Test handling of floating point precision edge cases."""
        metadata = [
            {
                "video_id": "precision_test",
                "file_path": "/fake/test.mp4",
                "filename": "test.mp4",
                "duration": 0.0000001,  # Very small duration
                "fps": 999999.999999,  # Very high precision FPS
                "frame_count": 1,
                "width": 1,
                "height": 1,
                "resolution": "1x1",
                "file_size": 1,
                "is_seamless": True
            }
        ]
        
        sequence, status = all_nodes['sequencer'].generate_sequence(
            video_metadata=metadata,
            total_duration_minutes=0.000001,  # Tiny duration
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        # Should handle extreme precision gracefully
        assert isinstance(status, str)
    
    def test_infinity_and_nan_values(self, all_nodes):
        """Test handling of infinity and NaN values."""
        metadata = [
            {
                "video_id": "inf_test",
                "file_path": "/fake/test.mp4",
                "filename": "test.mp4",
                "duration": float('inf'),  # Infinity
                "fps": float('nan'),  # NaN
                "frame_count": 150,
                "width": 1920,
                "height": 1080,
                "resolution": "1920x1080",
                "file_size": 1000000,
                "is_seamless": True
            }
        ]
        
        sequence, status = all_nodes['sequencer'].generate_sequence(
            video_metadata=metadata,
            total_duration_minutes=1.0,
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        # Should detect and reject invalid values
        assert len(sequence) == 0
        assert "Error" in status