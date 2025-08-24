---
name: Bug Report
about: Create a report to help us improve LoopyComfy
title: '[BUG] '
labels: bug
assignees: ''
---

## 🐛 Bug Report

### **Describe the Bug**
A clear and concise description of what the bug is.

### **To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

### **Expected Behavior**
A clear and concise description of what you expected to happen.

### **Screenshots**
If applicable, add screenshots to help explain your problem.

### **Environment Information**
Please fill out the following information:

**ComfyUI Environment:**
- ComfyUI Version: [e.g., v0.3.51]
- LoopyComfy Version: [e.g., v1.2.0]
- Installation Method: [Portable, Git Clone, Package Manager]

**System Information:**
- OS: [e.g., Windows 11, macOS 13, Ubuntu 22.04]
- Python Version: [e.g., 3.11.5]
- GPU: [e.g., RTX 3060, CPU only]
- RAM: [e.g., 16GB]

**Dependencies:**
- OpenCV Version: [run `python -c "import cv2; print(cv2.__version__)"`]
- NumPy Version: [run `python -c "import numpy; print(numpy.__version__)"`]
- FFmpeg Available: [Yes/No - run `ffmpeg -version`]

### **Node Configuration**
If applicable, provide the node configuration that caused the issue:

**VideoAssetLoader:**
- Directory path: [e.g., ./videos/]
- File pattern: [e.g., *.mp4]
- Max videos: [e.g., 10]
- Validate seamless: [Yes/No]

**MarkovVideoSequencer:**
- Total duration: [e.g., 30 seconds]
- Transition mode: [e.g., smart]
- Anti-repetition: [e.g., enabled]

**VideoSequenceComposer:**
- Resolution preset: [e.g., 1920×1080]
- Output FPS: [e.g., 30]
- Memory mode: [e.g., efficient]

**VideoSaver:**
- Platform preset: [e.g., YouTube]
- Output format: [e.g., mp4]
- Quality override: [e.g., auto]

### **Error Messages/Logs**
If applicable, paste the complete error message or relevant log output:

```
Paste error messages or logs here
```

### **ComfyUI Console Output**
If the issue occurs in ComfyUI, please include the console output:

```
Paste ComfyUI console output here
```

### **Additional Context**
Add any other context about the problem here.

### **Workaround**
If you found a temporary workaround, please describe it here.

---

## 🔍 **For LoopyComfy Developers**

### **Triage Checklist**
- [ ] Reproducible on clean ComfyUI installation
- [ ] Affects multiple operating systems
- [ ] Related to recent code changes
- [ ] Memory or performance related
- [ ] ComfyUI compatibility issue
- [ ] Mathematical/algorithm issue
- [ ] UI/UX enhancement issue

### **Priority Assessment**
- [ ] Critical (Crashes, data loss, security)
- [ ] High (Major feature broken, performance issue)
- [ ] Medium (Minor feature issue, usability problem)
- [ ] Low (Cosmetic, enhancement request)

### **Component Labels**
- [ ] `video-asset-loader`
- [ ] `markov-sequencer` 
- [ ] `video-composer`
- [ ] `video-saver`
- [ ] `markov-engine`
- [ ] `ui-enhancements`
- [ ] `documentation`
- [ ] `testing`
- [ ] `installation`