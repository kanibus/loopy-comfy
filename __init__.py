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

# ENHANCED: Robust import with detailed diagnostics and dependency checking
_IMPORTS_SUCCESSFUL = False
_IMPORT_ERRORS = []

def check_dependencies():
    """Check for required dependencies and provide detailed diagnostics."""
    missing_deps = []
    
    # Test critical dependencies
    critical_deps = [
        ('numpy', 'numpy'),
        ('cv2', 'opencv-python'), 
        ('psutil', 'psutil'),
        ('scipy', 'scipy'),
        ('PIL', 'Pillow'),
        ('imageio', 'imageio'),
        ('ffmpeg', 'ffmpeg-python')
    ]
    
    for dep_name, pip_name in critical_deps:
        try:
            __import__(dep_name)
        except ImportError:
            missing_deps.append(pip_name)
    
    return missing_deps

# Check dependencies first
missing_deps = check_dependencies()
if missing_deps:
    print("=" * 60)
    print("LOOPY-COMFY DEPENDENCY ERROR")
    print("=" * 60)
    print(f"Missing required dependencies: {', '.join(missing_deps)}")
    print("\nTo fix this issue, run:")
    print(f"pip install {' '.join(missing_deps)}")
    print("\nOr install all dependencies:")
    print("pip install -r requirements.txt")
    print("=" * 60)

# Try importing nodes with detailed error reporting
try:
    # Test each node individually for better error reporting
    print("Loading LoopyComfy nodes...")
    
    from nodes.video_asset_loader import LoopyComfy_VideoAssetLoader
    print("- VideoAssetLoader: OK")
    
    from nodes.markov_sequencer import LoopyComfy_MarkovVideoSequencer  
    print("- MarkovVideoSequencer: OK")
    
    from nodes.video_composer import LoopyComfy_VideoSequenceComposer
    print("- VideoSequenceComposer: OK")
    
    from nodes.video_saver import LoopyComfy_VideoSaver
    print("- VideoSaver: OK")
    
    _IMPORTS_SUCCESSFUL = True
    print("SUCCESS: Loopy-Comfy nodes loaded!")
    
except ImportError as e:
    print("=" * 60)
    print("LOOPY-COMFY NODE IMPORT ERROR")
    print("=" * 60)
    print(f"Error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {current_dir}")
    print("\nTroubleshooting steps:")
    print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
    print("2. Check that psutil is installed: pip install psutil>=5.9.6") 
    print("3. Restart ComfyUI after installing dependencies")
    print("4. Check the COMFYUI_INSTALLATION.md file for detailed instructions")
    print("=" * 60)
    _IMPORTS_SUCCESSFUL = False
    _IMPORT_ERRORS.append(str(e))
    
    # Create minimal working dummy classes that won't break ComfyUI
    class LoopyComfy_VideoAssetLoader:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"error_info": ("STRING", {"default": "Node failed to load - check console"})}}
        
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("error",)
        FUNCTION = "show_error"
        CATEGORY = "loopy-comfy"
        
        def show_error(self, error_info):
            return (f"Import error: {_IMPORT_ERRORS[0] if _IMPORT_ERRORS else 'Unknown error'}",)
    
    class LoopyComfy_MarkovVideoSequencer:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"error_info": ("STRING", {"default": "Node failed to load - check console"})}}
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("error",)
        FUNCTION = "show_error"
        CATEGORY = "loopy-comfy"
        def show_error(self, error_info):
            return (f"Import error: {_IMPORT_ERRORS[0] if _IMPORT_ERRORS else 'Unknown error'}",)
    
    class LoopyComfy_VideoSequenceComposer:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"error_info": ("STRING", {"default": "Node failed to load - check console"})}}
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("error",)
        FUNCTION = "show_error"
        CATEGORY = "loopy-comfy"
        def show_error(self, error_info):
            return (f"Import error: {_IMPORT_ERRORS[0] if _IMPORT_ERRORS else 'Unknown error'}",)
    
    class LoopyComfy_VideoSaver:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"error_info": ("STRING", {"default": "Node failed to load - check console"})}}
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("error",)
        FUNCTION = "show_error"
        CATEGORY = "loopy-comfy"
        def show_error(self, error_info):
            return (f"Import error: {_IMPORT_ERRORS[0] if _IMPORT_ERRORS else 'Unknown error'}",)

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

# Web directory for UI extensions
WEB_DIRECTORY = "./web"

__version__ = "1.0.0"
__author__ = "Loopy Comfy Team"
__description__ = "Dynamic non-repetitive video loops with Markov chains"