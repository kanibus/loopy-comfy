# ComfyUI Installation Fix - LoopyComfy Nodes

## Issue Identified ✅

The "Some Nodes Are Missing" error was caused by **missing psutil dependency** required for the PRPPRD security and performance features.

## Quick Fix Solution

### For ComfyUI Portable Installation:

1. **Open Command Prompt as Administrator**

2. **Navigate to your ComfyUI directory**:
   ```bash
   cd "I:\ComfyUI_windows_portable"
   ```

3. **Install missing dependency**:
   ```bash
   .\python_embeded\python.exe -m pip install psutil>=5.9.6
   ```

4. **Restart ComfyUI**:
   - Close ComfyUI completely
   - Restart ComfyUI server
   - Nodes should now load correctly

### For Standard ComfyUI Installation:

1. **Activate ComfyUI environment** (if using conda/venv)
2. **Install dependency**:
   ```bash
   pip install psutil>=5.9.6
   ```
3. **Restart ComfyUI**

## Verification Steps

After installation, verify the fix:

1. **Check ComfyUI console** - should see:
   ```
   Loading: custom_nodes\loopy-comfy
   ```

2. **Look for nodes in UI**:
   - 🎬 Video Asset Loader
   - 🎲 Markov Video Sequencer  
   - 🎞️ Video Sequence Composer
   - 💾 Video Saver

3. **If still having issues**:
   ```bash
   # Pull latest updates from GitHub
   cd "ComfyUI\custom_nodes\loopy-comfy"
   git pull origin main
   
   # Install all dependencies
   pip install -r requirements.txt
   ```

## What Was Fixed

- **Root Cause**: `psutil` library missing for memory/security monitoring
- **Files Affected**: `utils/memory_manager.py`, `utils/security_utils.py`
- **Solution**: Added `psutil>=5.9.6` to `requirements.txt`
- **Status**: ✅ **Fixed and pushed to GitHub**

## Complete Dependency List

The PRPPRD implementation requires:
- `numpy>=1.24.0` - Mathematical operations
- `opencv-python>=4.8.0` - Video processing  
- `ffmpeg-python>=0.2.0` - Video encoding
- `psutil>=5.9.6` - **NEW** - System monitoring
- `scipy>=1.10.0` - Scientific computing
- `scikit-learn>=1.3.0` - Machine learning
- `Pillow>=10.0.0` - Image processing
- `imageio>=2.30.0` - I/O operations

Your LoopyComfy nodes should now load successfully! 🚀