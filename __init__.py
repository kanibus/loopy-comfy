# -*- coding: utf-8 -*-
"""
Non-Linear Video Avatar - ComfyUI Custom Nodes

This package provides ComfyUI custom nodes for generating non-repetitive video loops
using Markov chain-based sequencing from a collection of video subloops.

ARCHITECTURAL ALIGNMENT: Implements PRP specification with LoopyComfy branding
- Node names follow PRP spec: NonLinearVideoAvatar_*
- Internal implementation remains LoopyComfy_* for backward compatibility
- Provides both naming conventions for maximum compatibility

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

# Import nodes with diagnostics but don't break functionality
_IMPORTS_SUCCESSFUL = False
_IMPORT_ERRORS = []

# Silent dependency check (only print errors, not success messages)
if missing_deps:
    print("LOOPY-COMFY: Missing dependencies detected. Run install_dependencies.py to fix.")

try:
    from nodes.video_asset_loader import LoopyComfy_VideoAssetLoader
    from nodes.markov_sequencer import LoopyComfy_MarkovVideoSequencer  
    from nodes.video_composer import LoopyComfy_VideoSequenceComposer
    from nodes.video_saver import LoopyComfy_VideoSaver
    
    _IMPORTS_SUCCESSFUL = True
    
except ImportError as e:
    print("=" * 60)
    print("LOOPY-COMFY IMPORT ERROR")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nQuick fix:")
    print("1. Run: python install_dependencies.py")
    print("2. Or manually: pip install psutil scipy scikit-learn")
    print("3. Restart ComfyUI")
    print("=" * 60)
    _IMPORTS_SUCCESSFUL = False
    _IMPORT_ERRORS.append(str(e))
    
    # CRITICAL FIX: Don't create dummy classes, re-raise the error so ComfyUI shows "missing nodes"
    # This prevents broken functionality and makes the issue clear
    raise ImportError(f"LoopyComfy nodes failed to load: {e}. Run install_dependencies.py to fix.")

# ARCHITECTURAL SOLUTION: Dual naming convention for maximum compatibility
# Primary node registration follows PRP specification: NonLinearVideoAvatar_*
# Secondary registration maintains LoopyComfy_* for backward compatibility

NODE_CLASS_MAPPINGS = {
    # PRIMARY: PRP-compliant node names (as specified in documentation)
    "NonLinearVideoAvatar_VideoAssetLoader": LoopyComfy_VideoAssetLoader,
    "NonLinearVideoAvatar_MarkovVideoSequencer": LoopyComfy_MarkovVideoSequencer,
    "NonLinearVideoAvatar_VideoSequenceComposer": LoopyComfy_VideoSequenceComposer,
    "NonLinearVideoAvatar_VideoSaver": LoopyComfy_VideoSaver,
    
    # SECONDARY: Backward compatibility for existing workflows
    "LoopyComfy_VideoAssetLoader": LoopyComfy_VideoAssetLoader,
    "LoopyComfy_MarkovVideoSequencer": LoopyComfy_MarkovVideoSequencer,
    "LoopyComfy_VideoSequenceComposer": LoopyComfy_VideoSequenceComposer,
    "LoopyComfy_VideoSaver": LoopyComfy_VideoSaver,
}

# Display names for ComfyUI menu (PRP-compliant primary names)
NODE_DISPLAY_NAME_MAPPINGS = {
    # PRIMARY: PRP specification names
    "NonLinearVideoAvatar_VideoAssetLoader": "🎬 Video Asset Loader",
    "NonLinearVideoAvatar_MarkovVideoSequencer": "🎲 Markov Video Sequencer", 
    "NonLinearVideoAvatar_VideoSequenceComposer": "🎞️ Video Sequence Composer",
    "NonLinearVideoAvatar_VideoSaver": "💾 Video Saver",
    
    # SECONDARY: Backward compatibility names
    "LoopyComfy_VideoAssetLoader": "🎬 Video Asset Loader (Legacy)",
    "LoopyComfy_MarkovVideoSequencer": "🎲 Markov Video Sequencer (Legacy)", 
    "LoopyComfy_VideoSequenceComposer": "🎞️ Video Sequence Composer (Legacy)",
    "LoopyComfy_VideoSaver": "💾 Video Saver (Legacy)",
}

# FIXED: ComfyUI Custom Type System Implementation
# ComfyUI requires custom types to be properly registered and serializable

# Custom type constants - these are the type identifiers used in node definitions
VIDEO_METADATA_LIST = "VIDEO_METADATA_LIST"
VIDEO_SEQUENCE = "VIDEO_SEQUENCE" 
TRANSITION_LOG = "TRANSITION_LOG"
STATISTICS = "STATISTICS"

# Type validation functions - ComfyUI calls these to validate type compatibility
def validate_video_metadata_list(value):
    """Validate VIDEO_METADATA_LIST type for ComfyUI."""
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, dict) and 'path' in item for item in value
    )

def validate_video_sequence(value):
    """Validate VIDEO_SEQUENCE type for ComfyUI."""
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, (str, dict)) for item in value
    )

def validate_transition_log(value):
    """Validate TRANSITION_LOG type for ComfyUI."""
    return isinstance(value, dict) and 'transitions' in value

def validate_statistics(value):
    """Validate STATISTICS type for ComfyUI."""
    return isinstance(value, dict)

# Register custom types with ComfyUI (if type registration system is available)
try:
    # Try to register with ComfyUI's type system
    import comfy.model_management as model_management
    if hasattr(model_management, 'register_custom_type'):
        model_management.register_custom_type(VIDEO_METADATA_LIST, validate_video_metadata_list)
        model_management.register_custom_type(VIDEO_SEQUENCE, validate_video_sequence)
        model_management.register_custom_type(TRANSITION_LOG, validate_transition_log)
        model_management.register_custom_type(STATISTICS, validate_statistics)
except (ImportError, AttributeError):
    # ComfyUI doesn't have custom type registration, proceed without it
    pass

# Web directory for UI extensions
WEB_DIRECTORY = "./web"

# ARCHITECTURAL METADATA: Aligned with PRP specifications
__version__ = "2.0.0-prpprd"
__author__ = "Non-Linear Video Avatar Team"
__description__ = "Non-Linear Video Avatar - Dynamic non-repetitive video loops with Markov chains"
__license__ = "MIT"
__project__ = "NonLinearVideoAvatar"
__prp_version__ = "2.0-CORRECTED"
__prpprd_phase__ = "PHASE_3_COMPLETED"

# Compatibility information
__legacy_names__ = ["LoopyComfy_*"]
__primary_names__ = ["NonLinearVideoAvatar_*"]
__backward_compatible__ = True