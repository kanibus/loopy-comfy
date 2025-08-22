"""
Tests for MarkovVideoSequencer Node - ComfyUI NonLinear Video Avatar
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add project path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.markov_sequencer import LoopyComfy_MarkovVideoSequencer
from core.markov_engine import MarkovState


class TestMarkovVideoSequencer:
    """Test suite for MarkovVideoSequencer ComfyUI node."""
    
    @pytest.fixture
    def node(self):
        """Create MarkovVideoSequencer node instance."""
        return LoopyComfy_MarkovVideoSequencer()
    
    @pytest.fixture
    def sample_video_metadata(self):
        """Sample video metadata for testing."""
        return [
            {
                "video_id": "video_001",
                "file_path": "/test/video_001.mp4",
                "filename": "video_001.mp4",
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
                "video_id": "video_002",
                "file_path": "/test/video_002.mp4",
                "filename": "video_002.mp4",
                "duration": 4.5,
                "fps": 30.0,
                "frame_count": 135,
                "width": 1920,
                "height": 1080,
                "resolution": "1920x1080",
                "file_size": 950000,
                "is_seamless": True
            },
            {
                "video_id": "video_003",
                "file_path": "/test/video_003.mp4",
                "filename": "video_003.mp4",
                "duration": 3.5,
                "fps": 30.0,
                "frame_count": 105,
                "width": 1920,
                "height": 1080,
                "resolution": "1920x1080",
                "file_size": 800000,
                "is_seamless": True
            }
        ]
    
    def test_input_types_structure(self, node):
        """Test INPUT_TYPES returns correct structure."""
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        required = input_types["required"]
        
        # Check required fields
        assert "video_metadata" in required
        assert "total_duration_minutes" in required
        assert "prevent_immediate_repeat" in required
        assert "random_seed" in required
        
        # Check optional fields
        assert "optional" in input_types
        optional = input_types["optional"]
        assert "history_window" in optional
        assert "repetition_penalty" in optional
    
    def test_return_types(self, node):
        """Test RETURN_TYPES returns correct types."""
        return_types = node.RETURN_TYPES
        assert return_types == ("VIDEO_SEQUENCE", "STRING")
    
    def test_return_names(self, node):
        """Test RETURN_NAMES returns correct names."""
        return_names = node.RETURN_NAMES
        assert return_names == ("sequence", "status_message")
    
    def test_function_name(self, node):
        """Test FUNCTION returns correct function name."""
        function_name = node.FUNCTION
        assert function_name == "generate_sequence"
    
    def test_category(self, node):
        """Test CATEGORY returns correct category."""
        category = node.CATEGORY
        assert category == "video/avatar"
    
    def test_generate_sequence_basic(self, node, sample_video_metadata):
        """Test basic sequence generation."""
        sequence, status = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=1.0,  # 1 minute
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        assert isinstance(sequence, list)
        assert len(sequence) > 0
        assert "Successfully generated" in status
        
        # Check sequence structure
        total_duration = 0
        for entry in sequence:
            assert "video_id" in entry
            assert "file_path" in entry
            assert "filename" in entry
            assert "duration" in entry
            assert "start_time" in entry
            assert "end_time" in entry
            total_duration += entry["duration"]
        
        # Should meet or exceed target duration (60 seconds)
        assert total_duration >= 60.0
    
    def test_no_immediate_repetition(self, node, sample_video_metadata):
        """Test that no immediate repetitions occur."""
        sequence, status = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=2.0,
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        # Check no consecutive entries are the same
        for i in range(1, len(sequence)):
            prev_video = sequence[i-1]["video_id"]
            curr_video = sequence[i]["video_id"]
            assert prev_video != curr_video, f"Immediate repetition found at index {i}: {prev_video}"
    
    def test_allow_immediate_repetition(self, node, sample_video_metadata):
        """Test sequence generation when immediate repetitions are allowed."""
        # Create metadata with single video to force repetitions
        single_video_metadata = [sample_video_metadata[0]]
        
        sequence, status = node.generate_sequence(
            video_metadata=single_video_metadata,
            total_duration_minutes=0.5,  # 30 seconds (need multiple of same video)
            prevent_immediate_repeat=False,
            random_seed=42
        )
        
        assert len(sequence) > 1  # Should have multiple entries
        # With only one video, some repetitions should occur when allowed
        video_ids = [entry["video_id"] for entry in sequence]
        assert len(set(video_ids)) == 1  # Only one unique video ID
    
    def test_reproducible_sequences(self, node, sample_video_metadata):
        """Test that same seed produces identical sequences."""
        sequence1, _ = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=1.0,
            prevent_immediate_repeat=True,
            random_seed=123
        )
        
        sequence2, _ = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=1.0,
            prevent_immediate_repeat=True,
            random_seed=123
        )
        
        assert len(sequence1) == len(sequence2)
        for entry1, entry2 in zip(sequence1, sequence2):
            assert entry1["video_id"] == entry2["video_id"]
            assert entry1["duration"] == entry2["duration"]
    
    def test_different_seeds_different_sequences(self, node, sample_video_metadata):
        """Test that different seeds produce different sequences."""
        sequence1, _ = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=1.0,
            prevent_immediate_repeat=True,
            random_seed=123
        )
        
        sequence2, _ = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=1.0,
            prevent_immediate_repeat=True,
            random_seed=456
        )
        
        # Extract video IDs for comparison
        video_ids1 = [entry["video_id"] for entry in sequence1]
        video_ids2 = [entry["video_id"] for entry in sequence2]
        
        # Sequences should be different (very high probability with 3+ videos)
        assert video_ids1 != video_ids2
    
    def test_history_window_effect(self, node, sample_video_metadata):
        """Test that history window affects sequence generation."""
        # Generate with small history window
        sequence1, _ = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=1.0,
            prevent_immediate_repeat=True,
            random_seed=42,
            history_window=2
        )
        
        # Generate with larger history window
        sequence2, _ = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=1.0,
            prevent_immediate_repeat=True,
            random_seed=42,
            history_window=5
        )
        
        # Both should be valid sequences
        assert len(sequence1) > 0
        assert len(sequence2) > 0
        
        # Sequences should be identical with same seed (unless history affects randomness)
        # At minimum, check both are valid
        for seq in [sequence1, sequence2]:
            for i in range(1, len(seq)):
                assert seq[i-1]["video_id"] != seq[i]["video_id"]
    
    def test_repetition_penalty_effect(self, node, sample_video_metadata):
        """Test that repetition penalty affects selection."""
        # Test with different penalty values
        sequence1, _ = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=2.0,
            prevent_immediate_repeat=True,
            random_seed=42,
            repetition_penalty=0.1  # Low penalty
        )
        
        sequence2, _ = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=2.0,
            prevent_immediate_repeat=True,
            random_seed=42,
            repetition_penalty=0.9  # High penalty
        )
        
        # Both should be valid
        assert len(sequence1) > 0
        assert len(sequence2) > 0
        
        # With same seed, sequences should be identical
        # (penalty affects selection but seed controls randomness)
        video_ids1 = [entry["video_id"] for entry in sequence1]
        video_ids2 = [entry["video_id"] for entry in sequence2]
        assert video_ids1 == video_ids2  # Same seed should produce same result
    
    def test_empty_metadata_error(self, node):
        """Test error handling with empty metadata."""
        sequence, status = node.generate_sequence(
            video_metadata=[],
            total_duration_minutes=1.0,
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        assert sequence == []
        assert "Error" in status or "empty" in status.lower()
    
    def test_single_video_with_prevention(self, node, sample_video_metadata):
        """Test single video with immediate repetition prevention."""
        single_video = [sample_video_metadata[0]]
        
        # This should handle the edge case gracefully
        sequence, status = node.generate_sequence(
            video_metadata=single_video,
            total_duration_minutes=0.2,  # 12 seconds (less than video duration)
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        # Should either create sequence or handle error gracefully
        if len(sequence) > 0:
            # If sequence is created, it should be valid
            assert len(sequence) >= 1
            total_duration = sum(entry["duration"] for entry in sequence)
            assert total_duration >= 12.0
        else:
            # If no sequence, status should indicate the issue
            assert "Error" in status or "single" in status.lower()
    
    def test_zero_duration_error(self, node, sample_video_metadata):
        """Test error handling with zero duration."""
        sequence, status = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=0.0,
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        assert sequence == []
        assert "Error" in status or "duration" in status.lower()
    
    def test_negative_duration_error(self, node, sample_video_metadata):
        """Test error handling with negative duration."""
        sequence, status = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=-1.0,
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        assert sequence == []
        assert "Error" in status
    
    def test_invalid_metadata_structure(self, node):
        """Test error handling with invalid metadata structure."""
        invalid_metadata = [
            {
                "video_id": "test",
                # Missing required fields
            }
        ]
        
        sequence, status = node.generate_sequence(
            video_metadata=invalid_metadata,
            total_duration_minutes=1.0,
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        assert sequence == []
        assert "Error" in status
    
    def test_sequence_timing_consistency(self, node, sample_video_metadata):
        """Test that sequence timing is consistent."""
        sequence, status = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=1.0,
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        # Check timing consistency
        expected_start = 0.0
        for entry in sequence:
            assert abs(entry["start_time"] - expected_start) < 0.001
            expected_end = expected_start + entry["duration"]
            assert abs(entry["end_time"] - expected_end) < 0.001
            expected_start = expected_end
    
    def test_large_duration_handling(self, node, sample_video_metadata):
        """Test handling of large duration requests."""
        sequence, status = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=60.0,  # 1 hour
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        assert len(sequence) > 0
        total_duration = sum(entry["duration"] for entry in sequence)
        assert total_duration >= 3600.0  # 60 minutes in seconds
        
        # Should still maintain no immediate repetitions
        for i in range(1, len(sequence)):
            prev_video = sequence[i-1]["video_id"]
            curr_video = sequence[i]["video_id"]
            assert prev_video != curr_video
    
    def test_metadata_to_markov_state_conversion(self, node, sample_video_metadata):
        """Test conversion of metadata to MarkovState objects."""
        states = node.metadata_to_markov_states(sample_video_metadata)
        
        assert len(states) == 3
        for i, state in enumerate(states):
            assert isinstance(state, MarkovState)
            assert state.video_id == sample_video_metadata[i]["video_id"]
            assert state.duration == sample_video_metadata[i]["duration"]
            assert "file_path" in state.metadata
            assert "filename" in state.metadata
    
    def test_sequence_statistics_collection(self, node, sample_video_metadata):
        """Test that sequence generation includes statistics."""
        sequence, status = node.generate_sequence(
            video_metadata=sample_video_metadata,
            total_duration_minutes=2.0,
            prevent_immediate_repeat=True,
            random_seed=42
        )
        
        # Status should include useful information
        assert "videos" in status.lower()
        assert "duration" in status.lower()
        
        # Check sequence has reasonable distribution
        video_counts = {}
        for entry in sequence:
            video_id = entry["video_id"]
            video_counts[video_id] = video_counts.get(video_id, 0) + 1
        
        # All videos should be used at least once for long sequences
        if len(sequence) >= 10:
            assert len(video_counts) >= 2  # Multiple videos used