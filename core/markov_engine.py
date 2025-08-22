"""
Markov Chain Engine for Non-Repetitive Video Sequencing

Core mathematical engine that generates non-repetitive video sequences using
Markov chain transitions with history-based penalties.
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import deque


@dataclass
class MarkovState:
    """
    Represents a state in the Markov chain (corresponds to a video asset).
    
    Attributes:
        video_id: Unique identifier for the video
        duration: Duration of the video in seconds
        metadata: Additional metadata about the video
        last_used: Timestamp when state was last selected
    """
    video_id: str
    duration: float
    metadata: Dict[str, Any]
    last_used: Optional[float] = None


class MarkovTransitionEngine:
    """
    Core engine for generating non-repetitive video sequences using Markov chains.
    
    This engine maintains a transition matrix and history buffer to ensure
    that video sequences don't exhibit immediate repetitions while maintaining
    natural stochastic behavior.
    """
    
    def __init__(
        self,
        states: List[MarkovState],
        history_window: int = 5,
        repetition_penalty: float = 0.1,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Markov transition engine.
        
        Args:
            states: List of MarkovState objects representing available videos
            history_window: Number of recent states to track for penalties
            repetition_penalty: Penalty factor for recently used states (0.0-1.0)
            random_seed: Random seed for reproducible sequences
        """
        self.states = states
        self.state_count = len(states)
        self.history_window = history_window
        self.repetition_penalty = repetition_penalty
        
        # Validate minimum state count
        if self.state_count < 1:
            raise ValueError("At least 1 state is required for Markov chain")
        
        # Special handling for single state (cannot prevent self-loops)
        if self.state_count == 1:
            self.history_window = 0  # No history tracking needed
            self.repetition_penalty = 0.0  # No repetition penalty possible
        
        # Create state ID to index mapping
        self.state_to_index = {state.video_id: i for i, state in enumerate(states)}
        self.index_to_state = {i: state.video_id for i, state in enumerate(states)}
        
        # Initialize transition matrix (uniform by default)
        self.transition_matrix = self._initialize_matrix()
        
        # History tracking
        self.history_buffer = deque(maxlen=history_window)
        self.current_state = None
        
        # Random number generator
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Statistics
        self.transition_count = 0
        self.transition_log = []
    
    def _initialize_matrix(self, mode: str = "uniform") -> np.ndarray:
        """
        Initialize the transition matrix based on the specified mode.
        
        Args:
            mode: Transition mode ("uniform", "similarity", "learned", "custom")
            
        Returns:
            Normalized transition matrix
        """
        if mode == "uniform":
            # Create uniform transition matrix
            matrix = np.ones((self.state_count, self.state_count), dtype=np.float32)
            
            # Special case: single state (cannot prevent self-loops)
            if self.state_count == 1:
                return matrix  # Matrix remains [[1.0]]
            
            # Prevent self-loops (immediate repetition) for multiple states
            np.fill_diagonal(matrix, 0.0)
            
            # Normalize rows to sum to 1
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = matrix / row_sums
            
            return matrix
        
        else:
            raise NotImplementedError(f"Transition mode '{mode}' not implemented yet")
    
    def get_next_state(self, current_state_id: Optional[str] = None) -> str:
        """
        Select the next state based on transition probabilities and history penalties.
        
        Args:
            current_state_id: Current state ID (if None, uses internal state)
            
        Returns:
            Next state ID
            
        Raises:
            ValueError: If current state is invalid
        """
        if current_state_id is None:
            current_state_id = self.current_state
        
        if current_state_id is None:
            # Initial state selection (uniform random)
            next_index = random.randint(0, self.state_count - 1)
            next_state_id = self.index_to_state[next_index]
        else:
            # Get current state index
            if current_state_id not in self.state_to_index:
                raise ValueError(f"Invalid current state: {current_state_id}")
            
            current_index = self.state_to_index[current_state_id]
            
            # Get base transition probabilities
            base_probs = self.transition_matrix[current_index].copy()
            
            # Apply history-based penalties
            adjusted_probs = self._apply_history_penalties(base_probs)
            
            # Sample next state
            next_index = np.random.choice(self.state_count, p=adjusted_probs)
            next_state_id = self.index_to_state[next_index]
        
        # Update history and state
        self._update_history(current_state_id, next_state_id)
        self.current_state = next_state_id
        self.transition_count += 1
        
        return next_state_id
    
    def _apply_history_penalties(self, base_probs: np.ndarray) -> np.ndarray:
        """
        Apply history-based penalties to prevent recent repetitions.
        
        Args:
            base_probs: Base transition probabilities
            
        Returns:
            Adjusted probabilities with history penalties applied
        """
        adjusted_probs = base_probs.copy()
        
        # Apply penalties based on recency in history
        for i, state_id in enumerate(self.history_buffer):
            if state_id in self.state_to_index:
                state_index = self.state_to_index[state_id]
                
                # Calculate penalty based on recency (more recent = higher penalty)
                recency_factor = (len(self.history_buffer) - i) / len(self.history_buffer)
                penalty = self.repetition_penalty * recency_factor
                
                # Apply penalty
                adjusted_probs[state_index] *= (1.0 - penalty)
        
        # Renormalize to ensure probabilities sum to 1
        prob_sum = adjusted_probs.sum()
        if prob_sum > 0:
            adjusted_probs = adjusted_probs / prob_sum
        else:
            # Fallback to uniform if all probabilities are zero
            adjusted_probs = np.ones(self.state_count) / self.state_count
        
        return adjusted_probs
    
    def _update_history(self, current_state_id: Optional[str], next_state_id: str):
        """
        Update history buffer and transition log.
        
        Args:
            current_state_id: Current state ID
            next_state_id: Next state ID
        """
        # Add to history buffer
        if next_state_id:
            self.history_buffer.append(next_state_id)
        
        # Log transition
        transition_record = {
            "from_state": current_state_id,
            "to_state": next_state_id,
            "transition_number": self.transition_count,
            "history_size": len(self.history_buffer)
        }
        self.transition_log.append(transition_record)
    
    def generate_sequence(
        self, 
        target_duration_minutes: float,
        start_state_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a complete video sequence for the target duration.
        
        Args:
            target_duration_minutes: Target duration in minutes
            start_state_id: Optional starting state ID
            
        Returns:
            List of sequence entries with state IDs and durations
            
        Raises:
            ValueError: If target duration is invalid
        """
        if target_duration_minutes <= 0:
            raise ValueError("Target duration must be positive")
        
        target_duration_seconds = target_duration_minutes * 60.0
        sequence = []
        total_duration = 0.0
        
        # Set starting state
        if start_state_id:
            if start_state_id not in self.state_to_index:
                raise ValueError(f"Invalid start state: {start_state_id}")
            current_state_id = start_state_id
        else:
            current_state_id = None
        
        # Generate sequence until target duration is reached
        while total_duration < target_duration_seconds:
            # Get next state
            next_state_id = self.get_next_state(current_state_id)
            
            # Get state duration
            state_index = self.state_to_index[next_state_id]
            state_duration = self.states[state_index].duration
            
            # Add to sequence
            sequence_entry = {
                "state_id": next_state_id,
                "video_id": next_state_id,
                "duration": state_duration,
                "start_time": total_duration,
                "end_time": total_duration + state_duration,
                "sequence_index": len(sequence),
                "metadata": self.states[state_index].metadata
            }
            
            sequence.append(sequence_entry)
            total_duration += state_duration
            current_state_id = next_state_id
            
            # Safety check to prevent infinite loops
            if len(sequence) > 10000:  # Reasonable upper limit
                print(f"Warning: Sequence generation stopped at {len(sequence)} entries")
                break
        
        return sequence
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get engine statistics and performance metrics.
        
        Returns:
            Dictionary containing statistics
        """
        # Calculate state usage frequency
        state_usage = {}
        for record in self.transition_log:
            state_id = record["to_state"]
            if state_id:
                state_usage[state_id] = state_usage.get(state_id, 0) + 1
        
        # Calculate transition distribution
        total_transitions = len(self.transition_log)
        usage_distribution = {
            state_id: count / total_transitions 
            for state_id, count in state_usage.items()
        } if total_transitions > 0 else {}
        
        # Check for immediate repetitions
        immediate_repetitions = 0
        for i in range(1, len(self.transition_log)):
            prev_state = self.transition_log[i-1]["to_state"]
            curr_state = self.transition_log[i]["to_state"]
            if prev_state == curr_state:
                immediate_repetitions += 1
        
        statistics = {
            "total_transitions": total_transitions,
            "unique_states_used": len(state_usage),
            "state_usage_frequency": state_usage,
            "usage_distribution": usage_distribution,
            "immediate_repetitions": immediate_repetitions,
            "repetition_rate": immediate_repetitions / total_transitions if total_transitions > 0 else 0,
            "history_window_size": self.history_window,
            "repetition_penalty": self.repetition_penalty,
            "current_history": list(self.history_buffer)
        }
        
        return statistics
    
    def reset(self):
        """Reset the engine state for a new sequence generation."""
        self.history_buffer.clear()
        self.current_state = None
        self.transition_count = 0
        self.transition_log.clear()


def validate_no_repetition(
    engine: MarkovTransitionEngine, 
    test_iterations: int = 10000
) -> Dict[str, Any]:
    """
    Validate that the engine doesn't produce immediate repetitions.
    
    Args:
        engine: MarkovTransitionEngine to test
        test_iterations: Number of transitions to test
        
    Returns:
        Validation results dictionary
    """
    engine.reset()
    
    immediate_repetitions = 0
    total_transitions = 0
    
    current_state = None
    
    for _ in range(test_iterations):
        next_state = engine.get_next_state(current_state)
        
        if current_state == next_state:
            immediate_repetitions += 1
        
        current_state = next_state
        total_transitions += 1
    
    repetition_rate = immediate_repetitions / total_transitions if total_transitions > 0 else 0
    
    validation_results = {
        "test_iterations": test_iterations,
        "immediate_repetitions": immediate_repetitions,
        "repetition_rate": repetition_rate,
        "passed": immediate_repetitions == 0,
        "engine_stats": engine.get_statistics()
    }
    
    return validation_results