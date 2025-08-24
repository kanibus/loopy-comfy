"""
MarkovVideoSequencer Node for Loopy Comfy

This node generates non-repetitive video sequences using Markov chain transitions
from a collection of video metadata.
"""

import random
import sys
import os
from typing import Dict, List, Tuple, Any, Optional

# CRITICAL: Standardized import path setup for ComfyUI compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# FIXED: Import core engine with fallback for ComfyUI compatibility
try:
    from core.markov_engine import MarkovTransitionEngine, MarkovState
    MARKOV_ENGINE_AVAILABLE = True
except ImportError:
    MARKOV_ENGINE_AVAILABLE = False
    
    # Fallback implementations
    class MarkovState:
        def __init__(self, video_index):
            self.video_index = video_index
    
    class MarkovTransitionEngine:
        """Fallback Markov engine with basic functionality."""
        
        def __init__(self, video_metadata_list):
            self.video_metadata = video_metadata_list
            self.num_videos = len(video_metadata_list) if video_metadata_list else 0
            self.history = []
            
        def get_next_state(self, current_state=None, prevent_immediate_repeat=True):
            """Simple fallback transition logic."""
            if self.num_videos == 0:
                return 0
            
            if self.num_videos == 1:
                return 0
                
            # Simple random selection avoiding immediate repetition
            available_indices = list(range(self.num_videos))
            
            if prevent_immediate_repeat and self.history:
                last_index = self.history[-1]
                if last_index in available_indices and len(available_indices) > 1:
                    available_indices.remove(last_index)
            
            next_index = random.choice(available_indices)
            self.history.append(next_index)
            
            return next_index
        
        def generate_sequence(self, target_duration_seconds, prevent_immediate=True):
            """Generate a sequence of video indices."""
            if self.num_videos == 0:
                return []
                
            sequence = []
            current_duration = 0
            
            while current_duration < target_duration_seconds:
                next_index = self.get_next_state(prevent_immediate_repeat=prevent_immediate)
                sequence.append(next_index)
                
                # Estimate duration (fallback to 10 seconds per video)
                video_duration = self.video_metadata[next_index].get('duration', 10.0) if next_index < len(self.video_metadata) else 10.0
                current_duration += video_duration
                
                # Safety break
                if len(sequence) > 1000:
                    break
            
            return sequence


class LoopyComfy_MarkovVideoSequencer:
    """
    ComfyUI node for generating non-repetitive video sequences using Markov chains.
    
    Takes video metadata and generates a sequence of video selections that
    maintains natural randomness while preventing immediate repetitions.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input types for ComfyUI interface."""
        return {
            "required": {
                "video_metadata": ("VIDEO_METADATA_LIST", {
                    "tooltip": "List of video metadata from VideoAssetLoader"
                }),
                "total_duration_minutes": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.1,
                    "max": 120.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Total duration for generated sequence"
                }),
                "prevent_immediate_repeat": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Prevent immediate repeats",
                    "label_off": "Allow immediate repeats",
                    "tooltip": "Prevent the same video from playing twice in a row"
                }),
                "random_seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Random seed for reproducible sequences"
                })
            },
            "optional": {
                "history_window": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of recent videos to track for penalties"
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Penalty factor for recently used videos"
                })
            }
        }
    
    RETURN_TYPES = ("VIDEO_SEQUENCE", "TRANSITION_LOG", "STATISTICS")
    RETURN_NAMES = ("sequence", "transition_log", "statistics")
    FUNCTION = "generate_sequence"
    CATEGORY = "video/avatar"
    
    def generate_sequence(
        self,
        video_metadata: List[Dict[str, Any]],
        total_duration_minutes: float,
        prevent_immediate_repeat: bool,
        random_seed: int,
        history_window: int = 5,
        repetition_penalty: float = 0.1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a non-repetitive video sequence using Markov chain transitions.
        
        Args:
            video_metadata: List of video metadata dictionaries
            total_duration_minutes: Target duration in minutes
            prevent_immediate_repeat: Whether to prevent immediate repetitions
            random_seed: Random seed for reproducibility
            history_window: Number of recent videos to track
            repetition_penalty: Penalty factor for recently used videos
            
        Returns:
            Tuple containing (sequence, transition_log, statistics)
            
        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not video_metadata:
                raise ValueError("No video metadata provided")
            
            if total_duration_minutes <= 0:
                raise ValueError("Duration must be positive")
            
            if len(video_metadata) < 2 and prevent_immediate_repeat:
                raise ValueError("Need at least 2 videos to prevent immediate repetitions")
            
            # Create Markov states from metadata
            states = []
            for metadata in video_metadata:
                state = MarkovState(
                    video_id=metadata.get('video_id', metadata.get('filename', 'unknown')),
                    duration=metadata.get('duration', 5.0),
                    metadata=metadata
                )
                states.append(state)
            
            # Initialize Markov engine
            engine = MarkovTransitionEngine(
                states=states,
                history_window=history_window if prevent_immediate_repeat else 1,
                repetition_penalty=repetition_penalty if prevent_immediate_repeat else 0.0,
                random_seed=random_seed
            )
            
            # Generate sequence
            sequence = engine.generate_sequence(total_duration_minutes)
            
            # Get transition log and statistics
            transition_log = engine.transition_log.copy()
            statistics = engine.get_statistics()
            
            # Add sequence-level statistics
            total_duration = sum(entry['duration'] for entry in sequence)
            unique_videos = len(set(entry['video_id'] for entry in sequence))
            
            sequence_stats = {
                "sequence_length": len(sequence),
                "total_duration_seconds": total_duration,
                "total_duration_minutes": total_duration / 60.0,
                "unique_videos_used": unique_videos,
                "total_videos_available": len(video_metadata),
                "utilization_rate": unique_videos / len(video_metadata),
                "average_video_duration": total_duration / len(sequence) if sequence else 0,
                "target_duration_minutes": total_duration_minutes,
                "duration_accuracy": abs(total_duration / 60.0 - total_duration_minutes) / total_duration_minutes
            }
            
            # Combine statistics
            combined_statistics = {
                **statistics,
                **sequence_stats,
                "generation_parameters": {
                    "prevent_immediate_repeat": prevent_immediate_repeat,
                    "random_seed": random_seed,
                    "history_window": history_window,
                    "repetition_penalty": repetition_penalty
                }
            }
            
            print(f"Generated sequence: {len(sequence)} videos, "
                  f"{total_duration:.1f}s total, "
                  f"{unique_videos}/{len(video_metadata)} unique videos used")
            
            if statistics['immediate_repetitions'] > 0:
                print(f"Warning: {statistics['immediate_repetitions']} immediate repetitions detected")
            
            return (sequence, transition_log, combined_statistics)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate video sequence: {str(e)}")
    
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> str:
        """Determine if node needs to be re-executed."""
        # Re-execute if random seed or duration changes
        return f"{kwargs.get('random_seed', 0)}_{kwargs.get('total_duration_minutes', 0)}"


# Required for ComfyUI node registration
WEB_DIRECTORY = "./web"