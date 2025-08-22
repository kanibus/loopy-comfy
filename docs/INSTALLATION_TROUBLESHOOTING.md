# 🔧 Installation Troubleshooting Guide

## 🚨 Common Installation Problems & Solutions

### **Problem 1: Python Version Incompatibility**

**Error Messages:**
```
ERROR: Could not find a version that satisfies the requirement numpy==1.24.3
ERROR: Cannot import 'setuptools.build_meta'
BackendUnavailable: Cannot import 'setuptools.build_meta'
```

**Root Cause:** Using Python 3.13+ which lacks wheel distributions for scientific packages.

**Solutions (in order of preference):**

#### **Solution 1A: Use Python 3.11 (RECOMMENDED)**
```bash
# Check your Python version
python --version

# If not 3.11.x, install Python 3.11
# Windows: Download from python.org
# macOS: brew install python@3.11  
# Ubuntu: sudo apt install python3.11 python3.11-dev python3.11-venv
```

#### **Solution 1B: Use Conda (BEST FOR COMPLEX SYSTEMS)**
```bash
# Install Miniconda/Anaconda first, then:
conda create -n loopy-comfy python=3.11
conda activate loopy-comfy
conda install numpy opencv scipy scikit-learn pillow
pip install ffmpeg-python imageio imageio-ffmpeg
```

#### **Solution 1C: Use pyenv (Linux/macOS)**
```bash
# Install pyenv first, then:
pyenv install 3.11.8
pyenv local 3.11.8
python --version  # Should show 3.11.8
```

---

### **Problem 2: Missing Build Tools**

**Error Messages:**
```
Microsoft Visual C++ 14.0 is required
error: Microsoft Visual Studio 14.0 is required
Building wheel for numpy (setup.py) ... error
```

**Solutions by Platform:**

#### **Windows Solutions:**
```bash
# Option A: Install Visual Studio Build Tools (RECOMMENDED)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Install "C++ build tools" workload

# Option B: Install setuptools first
pip install setuptools>=65.0.0 wheel>=0.40.0

# Option C: Use conda (avoids compilation entirely)
conda install numpy opencv scipy
```

#### **macOS Solutions:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# If that fails, install Xcode from App Store
# Then try again
```

#### **Linux (Ubuntu/Debian) Solutions:**
```bash
# Install build dependencies
sudo apt update
sudo apt install build-essential python3-dev python3-pip
sudo apt install libopencv-dev python3-opencv

# For CentOS/RHEL:
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel opencv-devel
```

---

### **Problem 3: FFmpeg Missing or Not Found**

**Error Messages:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
ffmpeg-python requires ffmpeg to be installed
```

**Solutions by Platform:**

#### **Windows:**
```bash
# Option A: Chocolatey (RECOMMENDED)
choco install ffmpeg

# Option B: Scoop
scoop install ffmpeg

# Option C: Manual installation
# 1. Download from https://ffmpeg.org/download.html#build-windows
# 2. Extract to C:\ffmpeg
# 3. Add C:\ffmpeg\bin to PATH environment variable
```

#### **macOS:**
```bash
# Using Homebrew (RECOMMENDED)
brew install ffmpeg

# Using MacPorts
sudo port install ffmpeg
```

#### **Linux:**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
# or
sudo dnf install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg
```

---

### **Problem 4: Permission/Access Errors**

**Error Messages:**
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
Access is denied
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
```bash
# Option A: Use --user flag (RECOMMENDED)
pip install --user -r requirements.txt

# Option B: Use virtual environment (BEST PRACTICE)
python -m venv loopy_env
# Windows:
loopy_env\Scripts\activate
# Linux/macOS:
source loopy_env/bin/activate
pip install -r requirements.txt

# Option C: Run as administrator (NOT RECOMMENDED)
# Windows: Run Command Prompt as Administrator
# Linux/macOS: Use sudo (avoid this)
```

---

### **Problem 5: ComfyUI Integration Issues**

**Error Messages:**
```
Some Nodes Are Missing
ImportError: cannot import name 'LoopyComfy_VideoAssetLoader'
ModuleNotFoundError: No module named 'nodes'
```

**Solutions:**

#### **Check Installation Location:**
```bash
# Verify you're in the correct directory
cd ComfyUI/custom_nodes/loopy-comfy
pwd  # Should show path ending in custom_nodes/loopy-comfy
```

#### **Verify Node Registration:**
```bash
# Check if __init__.py exists and is correct
ls -la __init__.py

# Check Python can import the nodes
python -c "from nodes.video_asset_loader import LoopyComfy_VideoAssetLoader; print('SUCCESS')"
```

#### **Restart ComfyUI Properly:**
```bash
# Stop ComfyUI completely (Ctrl+C)
# Wait 5 seconds
# Restart ComfyUI
# Check console for import errors
```

---

### **Problem 6: Memory/Performance Issues**

**Error Messages:**
```
Out of memory
MemoryError: Unable to allocate array
Process killed
```

**Solutions:**
```bash
# Reduce batch size in Video Sequence Composer node
# Default: 10 → Try: 5 or 3

# Lower output resolution
# 4K → 1080p → 720p

# Limit number of input videos
# Set max_videos in Video Asset Loader to 50 or fewer

# Close other applications to free RAM
# Consider using swap file/virtual memory
```

---

## 🔄 **Step-by-Step Installation Process**

### **Method 1: Standard Installation (Python 3.11)**
```bash
# 1. Verify Python version
python --version  # Must be 3.11.x

# 2. Navigate to ComfyUI custom nodes
cd ComfyUI/custom_nodes/

# 3. Clone repository
git clone https://github.com/kanibus/loopy-comfy.git
cd loopy-comfy

# 4. Install dependencies (try each step if previous fails)
pip install setuptools wheel  # Build tools first
pip install -r requirements.txt  # Main installation

# 5. Verify installation
python -c "import numpy, cv2, ffmpeg; print('All dependencies OK')"

# 6. Restart ComfyUI
```

### **Method 2: Conda Installation (MOST RELIABLE)**
```bash
# 1. Install Miniconda/Anaconda if not already installed

# 2. Create environment
conda create -n loopy-comfy python=3.11
conda activate loopy-comfy

# 3. Install scientific packages via conda
conda install numpy opencv scipy scikit-learn pillow

# 4. Install remaining packages via pip
pip install ffmpeg-python imageio imageio-ffmpeg pytest pytest-cov

# 5. Navigate and clone
cd ComfyUI/custom_nodes/
git clone https://github.com/kanibus/loopy-comfy.git

# 6. Always activate environment before starting ComfyUI
conda activate loopy-comfy
# Then start ComfyUI
```

### **Method 3: Docker Installation (ADVANCED)**
```bash
# Create Dockerfile in loopy-comfy directory:
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-c", "print('Loopy Comfy ready')"]

# Build and run:
docker build -t loopy-comfy .
docker run -v $(pwd):/app loopy-comfy
```

---

## 🧪 **Verification Commands**

### **Test Python Environment:**
```bash
python --version                          # Should be 3.11.x
python -c "import sys; print(sys.path)"  # Check Python path
```

### **Test Core Dependencies:**
```bash
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"  
python -c "import ffmpeg; print('FFmpeg Python OK')"
```

### **Test FFmpeg System:**
```bash
ffmpeg -version     # Should show FFmpeg version
which ffmpeg        # Should show path to FFmpeg binary
```

### **Test ComfyUI Integration:**
```bash
cd ComfyUI/custom_nodes/loopy-comfy
python -c "from __init__ import NODE_CLASS_MAPPINGS; print(list(NODE_CLASS_MAPPINGS.keys()))"
```

---

## 🆘 **Getting Additional Help**

### **If Nothing Works:**

1. **Create a System Report:**
```bash
# Create a file with this information:
python --version
pip --version
pip list | grep -E "(numpy|opencv|ffmpeg)"
ffmpeg -version 2>&1 | head -5
echo $PATH
uname -a  # Linux/macOS
ver       # Windows
```

2. **Check ComfyUI Console:**
   - Look for red error messages when starting ComfyUI
   - Copy the full error traceback
   - Note which specific import is failing

3. **Try Minimal Installation:**
```bash
# Install only core requirements:
pip install numpy>=1.24.0 opencv-python>=4.8.0
python -c "import numpy, cv2; print('Core OK')"
```

### **Community Support:**
- **GitHub Issues**: [Report bugs](https://github.com/kanibus/loopy-comfy/issues)
- **ComfyUI Discord**: General ComfyUI help
- **Stack Overflow**: Tag `python`, `opencv`, `ffmpeg`

---

## 📊 **Success Indicators**

✅ **Installation Successful When:**
- `python --version` shows 3.11.x
- `pip list` shows all required packages
- `ffmpeg -version` works without error
- ComfyUI starts without import errors
- Nodes appear in `video/avatar` category
- Example workflows load without "Missing Nodes" error

✅ **Ready to Use When:**
- Can load basic_avatar_workflow.json
- Video Asset Loader node accepts directory input
- All 4 nodes connect properly in workflow
- Queue Prompt executes without Python errors