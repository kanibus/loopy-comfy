# -*- coding: utf-8 -*-
"""
Loopy Comfy - Custom Nodes for Dynamic Video Loops

This package provides ComfyUI custom nodes for generating non-repetitive video loops
using Markov chain-based sequencing from a collection of video subloops.

CRITICAL: This __init__.py uses absolute imports to work with ComfyUI's direct execution model.
ComfyUI executes this file directly, not as a package import, so relative imports fail.
"""

import sys
import os

# CRITICAL FIX: Add project root to Python path for absolute imports
# This ensures ComfyUI can find and import all node modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# FIXED: Absolute imports work in ComfyUI environment (replaced relative imports)
try:
    from nodes.video_asset_loader import LoopyComfy_VideoAssetLoader
    from nodes.markov_sequencer import LoopyComfy_MarkovVideoSequencer  
    from nodes.video_composer import LoopyComfy_VideoSequenceComposer
    from nodes.video_saver import LoopyComfy_VideoSaver
    
    # Import success flag for debugging
    _IMPORTS_SUCCESSFUL = True
    
except ImportError as e:
    # Fallback error handling for debugging
    print(f"LOOPY-COMFY IMPORT ERROR: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {current_dir}")
    _IMPORTS_SUCCESSFUL = False
    
    # Create dummy classes to prevent ComfyUI crashes during development
    class LoopyComfy_VideoAssetLoader:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}
    
    class LoopyComfy_MarkovVideoSequencer:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}
    
    class LoopyComfy_VideoSequenceComposer:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}
    
    class LoopyComfy_VideoSaver:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

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