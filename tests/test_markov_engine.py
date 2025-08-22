"""
Tests for Markov Engine - Critical validation for no-repetition guarantee.
"""

import pytest
import numpy as np
from core.markov_engine import MarkovTransitionEngine, MarkovState, validate_no_repetition


class TestMarkovEngine:
    """Test suite for Markov transition engine."""
    
    def test_markov_state_creation(self):
        """Test MarkovState dataclass creation."""
        state = MarkovState(
            video_id="test_001",
            duration=5.0,
            metadata={"width": 1920, "height": 1080}
        )
        
        assert state.video_id == "test_001"
        assert state.duration == 5.0
        assert state.metadata["width"] == 1920
        assert state.last_used is None
    
    def test_engine_initialization(self):
        """Test engine initialization with states."""
        states = [
            MarkovState("video_001", 5.0, {}),
            MarkovState("video_002", 4.5, {}),
            MarkovState("video_003", 5.5, {})
        ]
        
        engine = MarkovTransitionEngine(states, random_seed=42)
        
        assert engine.state_count == 3
        assert len(engine.state_to_index) == 3
        assert engine.transition_matrix.shape == (3, 3)
        
        # Check transition matrix properties
        assert np.allclose(engine.transition_matrix.sum(axis=1), 1.0)  # Rows sum to 1
        assert np.all(np.diag(engine.transition_matrix) == 0)  # No self-loops
    
    def test_uniform_transition_matrix(self):
        """Test uniform transition matrix generation."""
        states = [MarkovState(f"video_{i:03d}", 5.0, {}) for i in range(5)]
        engine = MarkovTransitionEngine(states)
        
        # All non-diagonal elements should be equal
        matrix = engine.transition_matrix
        expected_value = 1.0 / (len(states) - 1)  # Exclude diagonal
        
        for i in range(len(states)):
            for j in range(len(states)):
                if i != j:
                    assert abs(matrix[i, j] - expected_value) < 1e-6
                else:
                    assert matrix[i, j] == 0.0
    
    def test_no_immediate_repetition_small_scale(self):
        """Test no immediate repetition with small number of iterations."""
        states = [MarkovState(f"video_{i:03d}", 5.0, {}) for i in range(3)]
        engine = MarkovTransitionEngine(states, random_seed=42)
        
        immediate_repetitions = 0
        current_state = None
        
        for _ in range(1000):
            next_state = engine.get_next_state(current_state)
            
            if current_state == next_state:
                immediate_repetitions += 1
            
            current_state = next_state
        
        assert immediate_repetitions == 0, f"Found {immediate_repetitions} immediate repetitions"
    
    def test_no_immediate_repetition_large_scale(self):
        """Test no immediate repetition with 10,000 iterations (CRITICAL TEST)."""
        states = [MarkovState(f"video_{i:03d}", 5.0, {}) for i in range(10)]
        engine = MarkovTransitionEngine(states, random_seed=42)
        
        validation_results = validate_no_repetition(engine, test_iterations=10000)
        
        assert validation_results["passed"], \
            f"Immediate repetitions found: {validation_results['immediate_repetitions']}"
        
        assert validation_results["immediate_repetitions"] == 0, \
            "CRITICAL FAILURE: Immediate repetitions detected in 10K test"
    
    def test_sequence_generation(self):
        """Test complete sequence generation."""
        states = [
            MarkovState("video_001", 5.0, {"filename": "video_001.mp4"}),
            MarkovState("video_002", 4.5, {"filename": "video_002.mp4"}),
            MarkovState("video_003", 5.5, {"filename": "video_003.mp4"})
        ]
        
        engine = MarkovTransitionEngine(states, random_seed=42)
        sequence = engine.generate_sequence(target_duration_minutes=1.0)  # 1 minute
        
        assert len(sequence) > 0
        
        # Check sequence properties
        total_duration = sum(entry['duration'] for entry in sequence)
        assert total_duration >= 60.0  # Should meet or exceed target
        
        # Check no immediate repetitions in sequence
        for i in range(1, len(sequence)):
            prev_video = sequence[i-1]['video_id']
            curr_video = sequence[i]['video_id']
            assert prev_video != curr_video, f"Immediate repetition at index {i}: {prev_video}"
    
    def test_history_penalties(self):
        """Test that history penalties affect state selection."""
        states = [MarkovState(f"video_{i:03d}", 5.0, {}) for i in range(5)]
        engine = MarkovTransitionEngine(states, history_window=3, repetition_penalty=0.5, random_seed=42)
        
        # Fill history with specific states
        for state_id in ["video_000", "video_001", "video_002"]:
            engine.history_buffer.append(state_id)
        
        # Get base probabilities for first state
        base_probs = engine.transition_matrix[0].copy()
        
        # Apply penalties
        adjusted_probs = engine._apply_history_penalties(base_probs)
        
        # States in history should have lower probabilities
        for i, state_id in enumerate(["video_000", "video_001", "video_002"]):
            state_index = engine.state_to_index[state_id]
            assert adjusted_probs[state_index] < base_probs[state_index]
        
        # Probabilities should still sum to ~1
        assert abs(adjusted_probs.sum() - 1.0) < 1e-6
    
    def test_statistics_tracking(self):
        """Test statistics collection."""
        states = [MarkovState(f"video_{i:03d}", 5.0, {}) for i in range(3)]
        engine = MarkovTransitionEngine(states, random_seed=42)
        
        # Generate some transitions
        current_state = None
        for _ in range(100):
            current_state = engine.get_next_state(current_state)
        
        stats = engine.get_statistics()
        
        assert stats["total_transitions"] == 100
        assert stats["immediate_repetitions"] == 0
        assert stats["repetition_rate"] == 0.0
        assert len(stats["state_usage_frequency"]) > 0
        assert abs(sum(stats["usage_distribution"].values()) - 1.0) < 1e-6
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Single state (should raise error with repetition prevention)
        single_state = [MarkovState("video_001", 5.0, {})]
        
        with pytest.raises(ValueError):
            engine = MarkovTransitionEngine(single_state)
            engine.get_next_state("video_001")  # Would cause immediate repetition
        
        # Empty states list
        with pytest.raises(ValueError):
            MarkovTransitionEngine([])
        
        # Invalid target duration
        states = [MarkovState(f"video_{i:03d}", 5.0, {}) for i in range(3)]
        engine = MarkovTransitionEngine(states)
        
        with pytest.raises(ValueError):
            engine.generate_sequence(-1.0)  # Negative duration
    
    def test_reproducibility(self):
        """Test that sequences are reproducible with same seed."""
        states = [MarkovState(f"video_{i:03d}", 5.0, {}) for i in range(5)]
        
        # Generate two sequences with same seed
        engine1 = MarkovTransitionEngine(states, random_seed=123)
        sequence1 = engine1.generate_sequence(1.0)
        
        engine2 = MarkovTransitionEngine(states, random_seed=123)
        sequence2 = engine2.generate_sequence(1.0)
        
        # Sequences should be identical
        assert len(sequence1) == len(sequence2)
        for entry1, entry2 in zip(sequence1, sequence2):
            assert entry1['video_id'] == entry2['video_id']
    
    def test_different_seeds_produce_different_sequences(self):
        """Test that different seeds produce different sequences."""
        states = [MarkovState(f"video_{i:03d}", 5.0, {}) for i in range(5)]
        
        engine1 = MarkovTransitionEngine(states, random_seed=123)
        sequence1 = engine1.generate_sequence(1.0)
        
        engine2 = MarkovTransitionEngine(states, random_seed=456)
        sequence2 = engine2.generate_sequence(1.0)
        
        # Sequences should be different (very high probability)
        video_ids1 = [entry['video_id'] for entry in sequence1]
        video_ids2 = [entry['video_id'] for entry in sequence2]
        
        assert video_ids1 != video_ids2, "Different seeds produced identical sequences"


class TestNoRepetitionCritical:
    """Critical tests for the no-repetition guarantee."""
    
    def test_no_repetition_10k_iterations(self):
        """CRITICAL TEST: Validate no immediate repetitions over 10,000 iterations."""
        # Test with different state counts
        for state_count in [3, 5, 10, 20]:
            states = [MarkovState(f"video_{i:03d}", 5.0, {}) for i in range(state_count)]
            engine = MarkovTransitionEngine(states, random_seed=42)
            
            validation_results = validate_no_repetition(engine, test_iterations=10000)
            
            assert validation_results["passed"], \
                f"CRITICAL FAILURE: Immediate repetitions with {state_count} states: " \
                f"{validation_results['immediate_repetitions']}"
            
            print(f"✅ No repetitions confirmed for {state_count} states over 10K iterations")
    
    def test_distribution_uniformity(self):
        """Test that state selection follows roughly uniform distribution."""
        states = [MarkovState(f"video_{i:03d}", 5.0, {}) for i in range(5)]
        engine = MarkovTransitionEngine(states, random_seed=42)
        
        # Generate many transitions
        state_counts = {state.video_id: 0 for state in states}
        current_state = None
        
        iterations = 10000
        for _ in range(iterations):
            next_state = engine.get_next_state(current_state)
            state_counts[next_state] += 1
            current_state = next_state
        
        # Check distribution is roughly uniform (within 15% of expected)
        expected_count = iterations / len(states)
        tolerance = 0.15
        
        for state_id, count in state_counts.items():
            deviation = abs(count - expected_count) / expected_count
            assert deviation < tolerance, \
                f"State {state_id} used {count} times, expected ~{expected_count} " \
                f"(deviation: {deviation:.2%})"