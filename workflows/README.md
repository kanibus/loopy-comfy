# ComfyUI NonLinear Video Avatar - Example Workflows

This directory contains example ComfyUI workflow files demonstrating how to use the NonLinear Video Avatar nodes.

## Workflow Files

### 1. `basic_avatar_workflow.json`
**Quick Start - 5 Minute Avatar**

**Configuration:**
- **Duration**: 5 minutes
- **Resolution**: 1920x1080 (1080p)
- **Quality**: Standard (CRF 23)
- **Codec**: H.264 (maximum compatibility)
- **Processing Time**: ~1-2 minutes

**Use Cases:**
- YouTube intros
- Social media content
- Quick demonstrations
- Testing the system

**Assets Needed:**
- 10-20 video clips in `./assets/sample_videos/`
- Each clip 3-7 seconds long
- MP4 format recommended

---

### 2. `advanced_avatar_workflow.json`
**Professional - 30 Minute Avatar**

**Configuration:**
- **Duration**: 30 minutes  
- **Resolution**: 3840x2160 (4K)
- **Quality**: High (CRF 18)
- **Codec**: H.265 (efficient compression)
- **Processing Time**: ~3-5 minutes

**Use Cases:**
- Professional presentations
- Long-form content creation
- Streaming backgrounds
- Corporate communications

**Assets Needed:**
- 50-100 video clips in `./assets/avatar_collection/`
- Each clip 3-7 seconds long
- Consistent resolution and frame rate
- High quality source material

---

### 3. `ui_enhanced_workflow.json`
**UI Enhancement Demo - All New Features**

**Configuration:**
- **Duration**: 30 seconds (demo)
- **Resolution**: 1920x1080
- **Quality**: Standard (CRF 23)
- **UI Features**: All new enhancements enabled

**Demonstrates:**
- 📁 **Native OS Folder Browser**: Click "Browse Folder" for native dialog
- 🎬 **Format Dropdown**: Descriptive codec selection (e.g., "mp4 (H.264 - Universal)")  
- 🖼️ **Memory-Safe Preview**: 360p thumbnail preview with frame limit
- 📊 **Video Count Display**: Shows count of videos found in selected folder
- 💾 **Output Directory Browser**: Native dialog for output folder selection
- 📈 **Memory Monitoring**: Real-time memory usage tracking
- ⚠️ **Error Handling**: Graceful fallbacks when native dialogs unavailable

**Use Cases:**
- Testing new UI enhancements
- Demonstrating native folder browser functionality  
- Validating memory-safe preview features
- Training users on enhanced interface

**Assets Needed:**
- 10-20 video clips in any directory
- Use the folder browser to select your video directory
- Supports all common video formats (.mp4, .mov, .avi, .webm, .mkv)

---

## How to Use These Workflows

### Step 1: Prepare Your Video Assets
1. Create the asset directory referenced in the workflow
2. Add your seamless loop video clips
3. Ensure consistent format and quality

### Step 2: Load the Workflow
1. Open ComfyUI
2. Click "Load" and select the workflow `.json` file
3. The nodes will appear connected and configured

### Step 3: Customize Parameters
- **VideoAssetLoader**: Adjust directory path and validation settings
- **MarkovVideoSequencer**: Modify duration and random seed
- **VideoSequenceComposer**: Change resolution and batch size
- **VideoSaver**: Set output filename and quality

### Step 4: Execute
1. Click "Queue Prompt" in ComfyUI
2. Monitor progress in the console
3. Find your generated video in the output directory

---

## Parameter Optimization Guide

### For Better Performance:
- **Reduce batch_size** if running out of memory
- **Lower resolution** for faster processing  
- **Use H.264** instead of H.265 for speed
- **Increase CRF value** for smaller files

### For Better Quality:
- **Use H.265 codec** for better compression
- **Lower CRF value** (18-20) for higher quality
- **Higher resolution** for crisp output
- **More video assets** for better variety

### For Memory Constraints:
- **batch_size: 5** for 8GB systems
- **batch_size: 10** for 16GB systems  
- **batch_size: 20** for 32GB+ systems

---

## Troubleshooting

### Common Issues:

**"No videos found"**
- Check the directory path in VideoAssetLoader
- Verify video files match the file pattern (*.mp4)
- Ensure directory exists and has read permissions

**"Out of memory"**
- Reduce batch_size in VideoSequenceComposer
- Lower the output resolution
- Close other applications to free RAM

**"FFmpeg codec not found"**
- Install FFmpeg system-wide
- Try different codec (h264 vs h265)
- Check ComfyUI console for detailed errors

**"Seamless validation failed"**
- Disable seamless validation temporarily
- Check that first/last frames of videos match
- Use video editing software to fix loop points

---

## Creating Custom Workflows

You can modify these workflows or create new ones:

1. **Start with a basic workflow**
2. **Add/modify nodes** as needed
3. **Connect the data flow**: VideoAssetLoader → MarkovVideoSequencer → VideoSequenceComposer → VideoSaver
4. **Adjust parameters** for your specific use case
5. **Save as new workflow** file

### Node Connection Pattern:
```
VideoAssetLoader.VIDEO_METADATA_LIST → MarkovVideoSequencer.video_metadata
MarkovVideoSequencer.VIDEO_SEQUENCE → VideoSequenceComposer.sequence  
VideoSequenceComposer.IMAGE → VideoSaver.frames
```

---

For more detailed information, see the main [README.md](../README.md) and [documentation](../docs/).