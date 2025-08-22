# Non-Linear Video Avatar ComfyUI Custom Nodes PRP

**Version:** 2.0  
**Date:** October 27, 2023  
**Author:** AI Software Architect  
**Project Codename:** DynamicAvatar

---

## Goal

Create a set of ComfyUI custom nodes that generate 30-minute non-repetitive video avatars from ~100 5-second video subloops using Markov chain-based sequencing, preventing obvious looping patterns and enabling future integration with external inputs for responsive behavior.

### Deliverables
- 4 custom ComfyUI nodes: VideoAssetLoader, MarkovVideoSequencer, VideoSequenceComposer, VideoSaver
- Example workflows demonstrating 30-minute avatar generation
- Comprehensive test suite validating Markov logic and performance
- Documentation for artists and developers

---

## Why

Current avatar systems using short video loops become obviously repetitive within 1-2 minutes, breaking immersion and appearing mechanical. This system creates lifelike, unpredictable avatars that maintain viewer engagement for extended periods (30+ minutes) without apparent repetition.

### Success Checklist
- [ ] Generate 30-minute videos without user-perceptible repetition patterns
- [ ] Zero immediate subloop repetitions (A-A sequences) in 10,000 transitions
- [ ] Process 100 5-second clips into 30-minute video in < 5 minutes (RTX 3060)
- [ ] Memory usage stays under 8GB peak during processing
- [ ] Seamless visual transitions between all subloop boundaries
- [ ] 95% user preference vs simple loops in A/B testing
- [ ] All nodes integrate seamlessly with existing ComfyUI video nodes
- [ ] Reproducible sequences via seed parameter
- [ ] External input hook functional for future sensor integration

---

## What

### Core Features

#### F1: Video Asset Management
Load and catalog video subloops with metadata extraction for transition compatibility scoring.

**Acceptance Criteria:**
- Loads all common video formats (.mp4, .mov, .avi, .webm)
- Extracts duration, resolution, frame rate automatically
- Validates seamless loop points
- Handles 100+ videos efficiently

#### F2: Markov Chain Sequencing
Generate non-repetitive sequences using weighted state transitions based on visual similarity and temporal coherence.

**Acceptance Criteria:**
- Implements configurable transition probability matrix
- Prevents immediate repetition (A→A transitions)
- Supports external signal modulation
- Maintains state history for pattern avoidance

#### F3: Seamless Video Composition
Concatenate selected subloops into continuous output stream with frame-perfect transitions.

**Acceptance Criteria:**
- Zero frame drops or duplicates at boundaries
- Maintains consistent frame rate
- Handles resolution mismatches gracefully
- Supports batch processing for memory efficiency

#### F4: Output Generation
Save composed video with configurable quality and format settings.

**Acceptance Criteria:**
- Supports H.264/H.265 encoding
- Configurable bitrate and quality
- Generates preview thumbnails
- Provides progress callbacks

### Node Specifications

#### Node 1: NonLinearVideoAvatar_VideoAssetLoader

```python
class NonLinearVideoAvatar_VideoAssetLoader:
    """Load and validate video subloop assets from directory."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": "./video_loops",
                    "multiline": False
                }),
                "file_pattern": ("STRING", {
                    "default": "*.mp4",
                    "multiline": False
                }),
                "validate_seamless": ("BOOLEAN", {
                    "default": True
                }),
                "max_videos": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                })
            }
        }
    
    RETURN_TYPES = ("VIDEO_METADATA_LIST", "INT", "FLOAT")
    RETURN_NAMES = ("video_metadata", "video_count", "total_duration")
    FUNCTION = "load_video_assets"
    CATEGORY = "video/avatar"
    
    def load_video_assets(self, directory_path, file_pattern, 
                         validate_seamless, max_videos):
        """
        Returns:
            video_metadata: List of dicts containing:
                - path: str
                - duration: float (seconds)
                - fps: float
                - resolution: tuple (width, height)
                - first_frame_hash: str (for matching)
                - last_frame_hash: str (for matching)
            video_count: Total videos loaded
            total_duration: Sum of all video durations
        """
        # Implementation here
        pass
```

#### Node 2: NonLinearVideoAvatar_MarkovVideoSequencer

```python
class NonLinearVideoAvatar_MarkovVideoSequencer:
    """Generate video sequence using Markov chain transitions."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_metadata": ("VIDEO_METADATA_LIST",),
                "total_duration_minutes": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "display": "slider"
                }),
                "transition_mode": (["uniform", "visual_similarity", "learned", "custom"],),
                "prevent_immediate_repeat": ("BOOLEAN", {"default": True}),
                "history_window": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of previous states to consider"
                }),
                "random_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                })
            },
            "optional": {
                "external_input_signal": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "transition_matrix": ("TENSOR",),
            }
        }
    
    RETURN_TYPES = ("VIDEO_SEQUENCE", "TRANSITION_LOG", "STATISTICS")
    RETURN_NAMES = ("sequence", "transition_log", "stats")
    FUNCTION = "generate_sequence"
    CATEGORY = "video/avatar"
    
    def generate_sequence(self, video_metadata, total_duration_minutes, 
                         transition_mode, prevent_immediate_repeat,
                         history_window, random_seed, 
                         external_input_signal=0.0, 
                         transition_matrix=None):
        """
        Markov Chain Algorithm:
        
        1. Initialize transition probability matrix P[i,j]
           - Uniform: P[i,j] = 1/(n-1) for i≠j, 0 for i=j
           - Visual: P[i,j] = exp(-similarity(i,j)) / Z
           - Learned: Use provided transition_matrix
           
        2. State selection with history:
           - Current state: s_t
           - History: H = [s_{t-k}, ..., s_{t-1}]
           - Valid next states: S' = {s | s ∉ H[−history_window:]}
           - P(s_{t+1} | s_t, H) ∝ P[s_t, s_{t+1}] * penalty(s_{t+1}, H)
           
        3. External signal modulation:
           - P'[i,j] = P[i,j] * (1 + α * external_signal * relevance[j])
           - α = modulation strength (0.5 default)
        """
        # Implementation here
        pass
```

#### Node 3: NonLinearVideoAvatar_VideoSequenceComposer

```python
class NonLinearVideoAvatar_VideoSequenceComposer:
    """Compose video frames from sequence with memory-efficient processing."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": ("VIDEO_SEQUENCE",),
                "output_fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1
                }),
                "resolution": (["maintain", "1920x1080", "1280x720", "custom"],),
                "batch_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Videos to load simultaneously"
                }),
                "blend_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 30,
                    "tooltip": "Frames to blend at transitions"
                })
            },
            "optional": {
                "custom_width": ("INT", {"default": 1920, "min": 128, "max": 8192}),
                "custom_height": ("INT", {"default": 1080, "min": 128, "max": 8192}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("frames", "total_frames", "duration_seconds")
    FUNCTION = "compose_video"
    CATEGORY = "video/avatar"
    OUTPUT_NODE = False
    
    def compose_video(self, sequence, output_fps, resolution, 
                     batch_size, blend_frames, 
                     custom_width=1920, custom_height=1080):
        """
        Memory-efficient composition:
        1. Process videos in batches of 'batch_size'
        2. Stream frames to disk if total > memory_threshold
        3. Apply transition blending if blend_frames > 0
        
        Expected tensor shape: [batch, frames, height, width, channels]
        """
        # Implementation here
        pass
```

#### Node 4: NonLinearVideoAvatar_VideoSaver

```python
class NonLinearVideoAvatar_VideoSaver:
    """Save composed video with encoding options."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "output_filename": ("STRING", {
                    "default": "avatar_%timestamp%.mp4",
                    "multiline": False
                }),
                "codec": (["h264", "h265", "vp9", "prores"],),
                "quality": ("INT", {
                    "default": 23,
                    "min": 0,
                    "max": 51,
                    "tooltip": "CRF value (lower=better quality)"
                }),
                "output_directory": ("STRING", {
                    "default": "./output",
                    "multiline": False
                })
            },
            "optional": {
                "audio_path": ("STRING", {"default": ""}),
                "metadata": ("DICT",)
            }
        }
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("filepath", "file_size_mb")
    FUNCTION = "save_video"
    CATEGORY = "video/avatar"
    OUTPUT_NODE = True
    
    def save_video(self, frames, output_filename, codec, 
                  quality, output_directory, 
                  audio_path="", metadata=None):
        """Save video using ffmpeg-python or OpenCV."""
        # Implementation here
        pass
```

### Markov Chain Implementation Details

```python
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class MarkovState:
    """Represents a video subloop state in the Markov chain."""
    index: int
    video_path: str
    duration: float
    features: np.ndarray  # Visual features for similarity
    
class MarkovTransitionEngine:
    """
    Core Markov chain logic for video sequencing.
    
    Mathematical Foundation:
    - State space S = {s_0, s_1, ..., s_n} where each s_i is a video subloop
    - Transition matrix P where P[i,j] = P(X_{t+1}=j | X_t=i)
    - History penalty function h(s, H) to avoid patterns
    - External modulation function m(s, signal) for future responsiveness
    """
    
    def __init__(self, states: List[MarkovState], mode: str = "uniform"):
        self.states = states
        self.n_states = len(states)
        self.transition_matrix = self._initialize_matrix(mode)
        self.history = []
        self.max_history = 10
        
    def _initialize_matrix(self, mode: str) -> np.ndarray:
        """
        Initialize transition probability matrix based on mode.
        
        Modes:
        - uniform: Equal probability to all non-self states
        - visual_similarity: Weighted by inverse visual distance
        - learned: Placeholder for ML-based transition learning
        """
        P = np.zeros((self.n_states, self.n_states))
        
        if mode == "uniform":
            # Equal probability, no self-loops
            P = (1 - np.eye(self.n_states)) / (self.n_states - 1)
            
        elif mode == "visual_similarity":
            # Calculate visual similarity matrix
            for i in range(self.n_states):
                for j in range(self.n_states):
                    if i != j:
                        # Cosine similarity of visual features
                        similarity = np.dot(self.states[i].features, 
                                          self.states[j].features)
                        P[i, j] = np.exp(-2 * (1 - similarity))
                        
            # Normalize rows
            P = P / P.sum(axis=1, keepdims=True)
            
        return P
    
    def get_next_state(self, 
                       current_state: int,
                       prevent_immediate: bool = True,
                       history_window: int = 5,
                       external_signal: float = 0.0) -> int:
        """
        Select next state based on current state and constraints.
        
        Algorithm:
        1. Get base probabilities from transition matrix
        2. Apply immediate repetition constraint
        3. Apply history penalty for recent states
        4. Apply external signal modulation
        5. Sample from resulting distribution
        """
        # Get transition probabilities for current state
        probs = self.transition_matrix[current_state].copy()
        
        # Prevent immediate repetition
        if prevent_immediate:
            probs[current_state] = 0.0
            
        # Apply history penalty
        if history_window > 0 and len(self.history) > 0:
            recent_states = self.history[-history_window:]
            for state in recent_states:
                # Exponential decay penalty based on recency
                recency = len(recent_states) - recent_states.index(state)
                penalty = 0.5 ** (recency / history_window)
                probs[state] *= penalty
                
        # Apply external signal modulation (future feature)
        if external_signal != 0.0:
            # Placeholder for sensor-based modulation
            # Example: boost "active" states when motion detected
            modulation = 1.0 + 0.5 * external_signal
            # Apply to relevant states (would be defined by metadata)
            probs *= modulation
            
        # Normalize probabilities
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # Fallback to uniform if all probabilities are zero
            probs = np.ones(self.n_states) / self.n_states
            probs[current_state] = 0.0
            probs = probs / probs.sum()
            
        # Sample next state
        next_state = np.random.choice(self.n_states, p=probs)
        
        # Update history
        self.history.append(current_state)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        return next_state
    
    def generate_sequence(self, 
                         duration_seconds: float,
                         avg_clip_duration: float = 5.0) -> List[int]:
        """Generate full sequence for target duration."""
        sequence = []
        current_duration = 0.0
        current_state = np.random.randint(0, self.n_states)
        
        while current_duration < duration_seconds:
            sequence.append(current_state)
            current_duration += self.states[current_state].duration
            current_state = self.get_next_state(current_state)
            
        return sequence
```

### Memory Management Strategy

```python
class MemoryEfficientVideoProcessor:
    """
    Handles video processing with strict memory constraints.
    
    Strategy:
    1. Never load all videos into memory simultaneously
    2. Use rolling buffer for active videos
    3. Stream large outputs to disk
    4. Implement aggressive garbage collection
    """
    
    MAX_MEMORY_GB = 8.0
    BUFFER_SIZE = 10  # Maximum videos in memory
    
    def __init__(self):
        self.video_cache = {}  # LRU cache for loaded videos
        self.cache_order = []  # Track access order
        self.memory_usage = 0.0
        
    def load_video_batch(self, paths: List[str], indices: List[int]):
        """Load videos with memory management."""
        for path, idx in zip(paths, indices):
            if idx in self.video_cache:
                # Move to end (most recently used)
                self.cache_order.remove(idx)
                self.cache_order.append(idx)
            else:
                # Check memory before loading
                self._ensure_memory_available()
                
                # Load video
                video_data = self._load_video_frames(path)
                self.video_cache[idx] = video_data
                self.cache_order.append(idx)
                
                # Update memory tracking
                self.memory_usage += self._estimate_memory(video_data)
                
    def _ensure_memory_available(self):
        """Free memory if needed."""
        while (self.memory_usage > self.MAX_MEMORY_GB * 0.9 or 
               len(self.video_cache) >= self.BUFFER_SIZE):
            if not self.cache_order:
                break
                
            # Remove least recently used
            lru_idx = self.cache_order.pop(0)
            video_data = self.video_cache.pop(lru_idx)
            self.memory_usage -= self._estimate_memory(video_data)
            
            # Force garbage collection
            import gc
            gc.collect()
```

---

## Context

### Critical References

```yaml
references:
  - url: https://github.com/comfyanonymous/ComfyUI/wiki/How-to-write-Custom-Nodes
    why: Official ComfyUI node development patterns and requirements
    
  - file: ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/videohelpersuite/nodes.py
    why: Reference implementation for video node patterns
    
  - doc: ComfyUI Execution Model
    section: Node execution order and data flow
    why: Understanding DAG execution for proper sequencing
    
  - file: ComfyUI/nodes.py
    section: LoadImage, SaveImage classes
    why: Pattern for file I/O operations in ComfyUI
    
  - url: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    why: OpenCV video processing patterns for Python
    
  - paper: "Hidden Markov Models for Video Generation"
    doi: 10.1145/3341156
    why: Mathematical foundation for video sequence generation

known_gotchas:
  - "ComfyUI requires @classmethod for INPUT_TYPES()"
  - "Video tensors must be shape [batch, frames, height, width, channels]"
  - "OpenCV uses BGR, ComfyUI expects RGB - must convert"
  - "Memory leaks with cv2.VideoCapture if not released properly"
  - "ComfyUI caches node outputs - design for deterministic behavior"
  - "Large tensors (>2GB) can crash ComfyUI without proper batching"
  - "File paths must be absolute or relative to ComfyUI root"
  - "Custom types need registration in __init__.py"
  
code_patterns:
  - pattern: "Node class naming"
    example: "ClassName_NodeName for menu organization"
    
  - pattern: "Error handling in nodes"
    example: |
      try:
          result = process()
      except Exception as e:
          print(f"Error in {self.__class__.__name__}: {e}")
          return (None,)  # Must return tuple
          
  - pattern: "Progress reporting"
    example: |
      from comfy.utils import ProgressBar
      pbar = ProgressBar(total_steps)
      pbar.update(current_step)
```

### Example Workflow JSON

```json
{
  "version": "1.0",
  "nodes": [
    {
      "id": 1,
      "type": "NonLinearVideoAvatar_VideoAssetLoader",
      "pos": [100, 100],
      "size": [315, 120],
      "mode": 0,
      "inputs": [],
      "outputs": [
        {"name": "video_metadata", "type": "VIDEO_METADATA_LIST"},
        {"name": "video_count", "type": "INT"},
        {"name": "total_duration", "type": "FLOAT"}
      ],
      "properties": {
        "directory_path": "./assets/avatar_loops",
        "file_pattern": "*.mp4",
        "validate_seamless": true,
        "max_videos": 100
      }
    },
    {
      "id": 2,
      "type": "NonLinearVideoAvatar_MarkovVideoSequencer",
      "pos": [450, 100],
      "size": [350, 200],
      "mode": 0,
      "inputs": [
        {"name": "video_metadata", "type": "VIDEO_METADATA_LIST", "link": 1}
      ],
      "outputs": [
        {"name": "sequence", "type": "VIDEO_SEQUENCE"},
        {"name": "transition_log", "type": "TRANSITION_LOG"},
        {"name": "stats", "type": "STATISTICS"}
      ],
      "properties": {
        "total_duration_minutes": 30,
        "transition_mode": "visual_similarity",
        "prevent_immediate_repeat": true,
        "history_window": 5,
        "random_seed": 42
      }
    },
    {
      "id": 3,
      "type": "NonLinearVideoAvatar_VideoSequenceComposer",
      "pos": [850, 100],
      "size": [320, 180],
      "mode": 0,
      "inputs": [
        {"name": "sequence", "type": "VIDEO_SEQUENCE", "link": 2}
      ],
      "outputs": [
        {"name": "frames", "type": "IMAGE"},
        {"name": "total_frames", "type": "INT"},
        {"name": "duration_seconds", "type": "FLOAT"}
      ],
      "properties": {
        "output_fps": 30.0,
        "resolution": "1920x1080",
        "batch_size": 10,
        "blend_frames": 3
      }
    },
    {
      "id": 4,
      "type": "NonLinearVideoAvatar_VideoSaver",
      "pos": [1200, 100],
      "size": [300, 160],
      "mode": 0,
      "inputs": [
        {"name": "frames", "type": "IMAGE", "link": 3}
      ],
      "outputs": [
        {"name": "filepath", "type": "STRING"},
        {"name": "file_size_mb", "type": "INT"}
      ],
      "properties": {
        "output_filename": "avatar_dynamic.mp4",
        "codec": "h264",
        "quality": 23,
        "output_directory": "./output"
      }
    }
  ],
  "links": [
    [1, 1, 0, 2, 0, "VIDEO_METADATA_LIST"],
    [2, 2, 0, 3, 0, "VIDEO_SEQUENCE"],
    [3, 3, 0, 4, 0, "IMAGE"]
  ]
}
```

### Validation Gates

```python
# validation_suite.py

class ValidationGates:
    """Progressive validation ensuring correctness at each level."""
    
    @staticmethod
    def level_1_syntax():
        """Validate Python syntax and imports."""
        import ast
        import pylint.lint
        
        files = ["nodes/video_avatar_nodes.py"]
        for file in files:
            with open(file) as f:
                ast.parse(f.read())  # Syntax check
                
        # Style check
        pylint.lint.Run(files + ['--disable=all', '--enable=E'])
        
    @staticmethod
    def level_2_node_registration():
        """Verify nodes register correctly in ComfyUI."""
        from nodes.video_avatar_nodes import NODE_CLASS_MAPPINGS
        
        required_nodes = [
            "NonLinearVideoAvatar_VideoAssetLoader",
            "NonLinearVideoAvatar_MarkovVideoSequencer",
            "NonLinearVideoAvatar_VideoSequenceComposer",
            "NonLinearVideoAvatar_VideoSaver"
        ]
        
        for node in required_nodes:
            assert node in NODE_CLASS_MAPPINGS
            assert hasattr(NODE_CLASS_MAPPINGS[node], "INPUT_TYPES")
            assert hasattr(NODE_CLASS_MAPPINGS[node], "FUNCTION")
            
    @staticmethod
    def level_3_markov_logic():
        """Test Markov chain logic correctness."""
        engine = MarkovTransitionEngine(test_states, mode="uniform")
        
        # Test no immediate repetition
        repetitions = 0
        last_state = -1
        for _ in range(10000):
            next_state = engine.get_next_state(
                np.random.randint(0, 10),
                prevent_immediate=True
            )
            if next_state == last_state:
                repetitions += 1
            last_state = next_state
            
        assert repetitions == 0, f"Found {repetitions} immediate repetitions"
        
    @staticmethod
    def level_4_memory_management():
        """Verify memory stays within limits."""
        import tracemalloc
        
        tracemalloc.start()
        
        # Simulate processing 100 videos
        processor = MemoryEfficientVideoProcessor()
        for i in range(100):
            processor.load_video_batch([f"video_{i}.mp4"], [i])
            
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_gb = peak / 1024 / 1024 / 1024
        assert peak_gb < 8.0, f"Peak memory {peak_gb:.2f}GB exceeds limit"
        
    @staticmethod  
    def level_5_end_to_end():
        """Full workflow execution test."""
        import json
        
        # Load test workflow
        with open("test_workflows/30_minute_avatar.json") as f:
            workflow = json.load(f)
            
        # Execute in ComfyUI
        from comfy.execution import execute_workflow
        result = execute_workflow(workflow)
        
        # Verify output
        assert result["success"]
        assert result["duration_seconds"] >= 1800  # 30 minutes
        assert os.path.exists(result["output_path"])
        
        # Verify no repetitions in sequence
        sequence = result["sequence_log"]
        for i in range(1, len(sequence)):
            assert sequence[i] != sequence[i-1], "Immediate repetition found"
```

### Error Handling Specifications

```python
ERROR_HANDLING = {
    "missing_videos": {
        "behavior": "Warning message, continue with found videos",
        "min_threshold": 10,  # Minimum videos needed
        "message": "Found only {count} videos, minimum 10 recommended"
    },
    
    "corrupt_video": {
        "behavior": "Skip silently, log to console",
        "fallback": "Use next valid video in sequence",
        "logging": "console.warning(f'Skipping corrupt video: {path}')"
    },
    
    "memory_overflow": {
        "behavior": "Automatic batch size reduction",
        "strategy": "Reduce batch_size by 50%, retry",
        "min_batch": 1,
        "message": "Reducing batch size due to memory constraints"
    },
    
    "incompatible_format": {
        "behavior": "Attempt conversion via ffmpeg",
        "fallback": "Skip if conversion fails", 
        "supported": [".mp4", ".mov", ".avi", ".webm", ".mkv"]
    },
    
    "resolution_mismatch": {
        "behavior": "Auto-scale to target resolution",
        "method": "Lanczos resampling",
        "maintain_aspect": True
    },
    
    "frame_rate_mismatch": {
        "behavior": "Interpolate to target fps",
        "method": "Motion-compensated interpolation",
        "fallback": "Frame duplication/dropping"
    }
}
```

### Installation & Setup

```bash
# Installation script
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-NonLinearVideoAvatar
cd ComfyUI-NonLinearVideoAvatar
pip install -r requirements.txt

# requirements.txt
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.10.0  # For similarity calculations
scikit-learn>=1.3.0  # For feature extraction
ffmpeg-python>=0.2.0  # For video encoding
tqdm>=4.65.0  # For progress bars
```

### Testing Data Structure

```python
# test_data.py
TEST_SCENARIOS = {
    "basic_30_minute": {
        "video_count": 100,
        "video_duration": 5.0,
        "target_duration": 1800,
        "expected_transitions": 360,
        "max_memory_gb": 8.0
    },
    
    "stress_test": {
        "video_count": 500,
        "video_duration": 3.0,
        "target_duration": 3600,
        "expected_transitions": 1200,
        "max_memory_gb": 8.0
    },
    
    "minimal_test": {
        "video_count": 10,
        "video_duration": 5.0,
        "target_duration": 60,
        "expected_transitions": 12,
        "max_memory_gb": 2.0
    }
}
```

---

## Appendix A: Mathematical Formulation

### Markov Chain Formal Definition

Let **S** = {s₁, s₂, ..., sₙ} be the state space where each state represents a video subloop.

The transition probability matrix **P** is defined as:
- P[i,j] = P(Xₜ₊₁ = sⱼ | Xₜ = sᵢ)
- Subject to: ∑ⱼ P[i,j] = 1 for all i
- Constraint: P[i,i] = 0 (no self-loops)

The history-aware transition probability:
- P'(sⱼ | sᵢ, H) = P[i,j] × ∏ₖ∈H penalty(sⱼ, sₖ, t-tₖ)
- Where penalty(sⱼ, sₖ, Δt) = exp(-λ/Δt) if sⱼ = sₖ, else 1
- λ = decay parameter (default 2.0)

External signal modulation:
- P''[i,j] = P'[i,j] × (1 + α × signal × relevance[j])
- α ∈ [0, 1] = modulation strength
- signal ∈ [-1, 1] = external input
- relevance[j] ∈ [0, 1] = pre-computed relevance score

---

## Appendix B: Future Enhancements

### Phase 2: External Input Integration (Q1 2024)
- Webcam gesture recognition node
- Audio reactivity node  
- Network message receiver node
- Sensor data interpreter node

### Phase 3: Advanced Sequencing (Q2 2024)
- LSTM-based sequence generation
- Reinforcement learning for optimal transitions
- Style transfer between subloops
- Emotional state modeling

### Phase 4: Real-time Preview (Q3 2024)
- WebRTC streaming node
- Live parameter adjustment
- Interactive sequence editing
- Performance profiling overlay

---

## Version History

- **v2.0** (Oct 27, 2023): Complete restructure following Wirasm/PRPs-agentic-eng best practices
- **v1.0** (Oct 26, 2023): Initial PRP draft

---

**END OF PRP DOCUMENT**