# Changelog

All notable changes to ComfyUI NonLinear Video Avatar will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Advanced video transition modes for future releases
- Real-time processing capabilities planning
- Enhanced ML integration roadmap

### Changed
- N/A

### Deprecated  
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [1.2.0] - 2024-12-21 - UI Enhancements & Bug Fixes Release

### Added
- **Native OS Folder Browser**: Tkinter-based folder selection dialogs for video input and output directories
- **FFmpeg Format Dropdown**: Descriptive format selection with codec information (e.g., "mp4 (H.264 - Universal)")
- **Memory-Safe Preview**: 360p preview with 150-frame limit and 1GB memory constraint enforcement
- **Server API Extensions**: `/loopycomfy/browse_folder` and `/loopycomfy/browse_output_dir` endpoints
- **Enhanced JavaScript UI**: New widgets for folder browsing with visual feedback and recent folder history
- **Memory Limit Enforcement**: Hard 8GB memory limit with real-time monitoring during video processing
- **Comprehensive GitHub Integration**: CI/CD pipeline, issue templates, PR templates
- **Modern Python Packaging**: pyproject.toml with full metadata and development dependencies
- **Video Count Display**: Shows number of videos found when selecting folders

### Changed
- **VideoAssetLoader**: Added `browse_folder` and `show_preview` parameters with backward compatibility
- **VideoSaver**: Added `browse_output_dir` parameter and enhanced format selection
- **VideoSequenceComposer**: Improved memory monitoring with warning thresholds
- **Security Enhancement**: Strengthened path validation with whitelist approach
- **Codec Validation**: Optimized FFmpeg codec testing for faster validation

### Fixed
- **CRITICAL**: Fixed function signature mismatch in VideoSequenceComposer `_process_video_batch()` method
- **SECURITY**: Fixed potential path traversal vulnerability in video asset loader
- **PERFORMANCE**: Optimized codec validation from 320x240x0.1s to 1x1x0.01s test pattern
- **MEMORY**: Added real-time memory limit enforcement during batch processing
- **UI**: Fixed Unicode character issues in test output for Windows compatibility

### Security
- **Path Security**: Enhanced directory validation with explicit whitelist of allowed paths
- **Input Sanitization**: Improved FFmpeg argument validation and metadata sanitization
- **Memory Protection**: Hard limits prevent memory exhaustion attacks

### Removed
- Unused code comments and development artifacts
- Legacy FFmpeg encoding method in favor of enhanced version

### Technical Details
- **Backward Compatibility**: All existing workflows continue to work unchanged
- **Test Coverage**: 200+ test cases with comprehensive validation including 10K no-repetition tests
- **Memory Safety**: All preview operations respect 1GB limits with graceful degradation
- **Cross-Platform**: Full Windows/macOS/Linux support with platform-specific optimizations
- **ComfyUI Integration**: Proper custom type definitions and node registration

## [1.1.0] - 2024-12-15

### Added
- **CRITICAL**: Production-ready ComfyUI node implementation (4 nodes)
- **CORE**: Advanced Markov chain engine with 10,000+ iteration no-repetition validation
- **QUALITY**: Comprehensive test suite with 148 test functions across 8 test files
- **PERFORMANCE**: Memory-efficient processing with 8GB constraint validation
- **INTEGRATION**: Complete ComfyUI custom node registration and type system
- **ENCODING**: Professional FFmpeg integration with multiple codec support (H.264/H.265)
- **WORKFLOWS**: Production-ready example workflows for 5-minute and 30-minute avatars
- **DOCUMENTATION**: Complete user guides, developer documentation, and troubleshooting

### Changed
- **Upgraded requirements.txt**: Added exact version constraints for production stability
- **Enhanced error handling**: Comprehensive validation and user feedback systems
- **Improved Unicode compatibility**: Fixed Windows terminal encoding issues
- **Optimized import resolution**: Absolute path handling for ComfyUI environment

### Fixed
- **CRITICAL**: Import path resolution issues in ComfyUI nodes
- **CRITICAL**: Single-state edge case handling in Markov engine
- **HIGH**: FFmpeg codec validation logic with proper test patterns
- **MEDIUM**: Thread-safe process monitoring with timeout handling
- **LOW**: Unicode compatibility across Windows/Linux platforms
- **SECURITY**: Path traversal vulnerability prevention in file operations

### Security
- **Path sanitization**: Prevented directory traversal attacks in file operations
- **Metadata validation**: FFmpeg parameter sanitization against injection
- **Process isolation**: Secure subprocess handling with timeout enforcement
- **Memory protection**: Monitored memory usage to prevent system instability

## [1.0.0] - 2023-10-27

### Added
- Project initialization
- Repository structure creation
- Core documentation files
- MIT license
- Basic project scaffolding

---

## Version History Template

Use the following template for future releases:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features added

### Changed  
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Now removed features

### Fixed
- Bug fixes

### Security
- Security vulnerability fixes
```

## Release Types

- **Major (X.0.0)**: Breaking changes, major new features
- **Minor (X.Y.0)**: New features, backward compatible
- **Patch (X.Y.Z)**: Bug fixes, backward compatible

## Categories

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Now removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes