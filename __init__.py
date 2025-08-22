"""
Loopy Comfy - Custom Nodes for Dynamic Video Loops

This package provides ComfyUI custom nodes for generating non-repetitive video loops
using Markov chain-based sequencing from a collection of video subloops.
"""

from .nodes.video_asset_loader import LoopyComfy_VideoAssetLoader
from .nodes.markov_sequencer import LoopyComfy_MarkovVideoSequencer  
from .nodes.video_composer import LoopyComfy_VideoSequenceComposer
from .nodes.video_saver import LoopyComfy_VideoSaver

# Node class mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LoopyComfy_VideoAssetLoader": LoopyComfy_VideoAssetLoader,
    "LoopyComfy_MarkovVideoSequencer": LoopyComfy_MarkovVideoSequencer,
    "LoopyComfy_VideoSequenceComposer": LoopyComfy_VideoSequenceComposer,
    "LoopyComfy_VideoSaver": LoopyComfy_VideoSaver,
}

# Display names for ComfyUI menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoopyComfy_VideoAssetLoader": "🎬 Video Asset Loader",
    "LoopyComfy_MarkovVideoSequencer": "🎲 Markov Video Sequencer", 
    "LoopyComfy_VideoSequenceComposer": "🎞️ Video Sequence Composer",
    "LoopyComfy_VideoSaver": "💾 Video Saver",
}

# Custom type definitions for ComfyUI
# These types allow proper data flow between nodes

class VideoMetadataList:
    """Container for video metadata collection"""
    def __init__(self, metadata_list):
        self.metadata_list = metadata_list
    
class VideoSequence:
    """Container for Markov-generated video sequence"""
    def __init__(self, sequence):
        self.sequence = sequence
        
class TransitionLog:
    """Container for transition statistics and logs"""
    def __init__(self, log_data):
        self.log_data = log_data
        
class Statistics:
    """Container for processing statistics"""
    def __init__(self, stats_data):
        self.stats_data = stats_data

# Custom type registrations for ComfyUI
VIDEO_METADATA_LIST = "VIDEO_METADATA_LIST"
VIDEO_SEQUENCE = "VIDEO_SEQUENCE"
TRANSITION_LOG = "TRANSITION_LOG" 
STATISTICS = "STATISTICS"

__version__ = "1.0.0"
__author__ = "Loopy Comfy Team"
__description__ = "Dynamic non-repetitive video loops with Markov chains"