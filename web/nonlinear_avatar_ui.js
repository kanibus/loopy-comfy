/**
 * Non-Linear Video Avatar - UI Extensions for ComfyUI
 * 
 * This file provides enhanced UI widgets for the Non-Linear Video Avatar nodes,
 * including folder browser functionality and resolution preset visualization.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Enhanced folder browser widget
class FolderBrowserWidget {
    constructor(node, inputName, inputData) {
        this.node = node;
        this.inputName = inputName;
        this.value = inputData[1]?.default || "";
        
        // Create main container
        this.element = document.createElement("div");
        this.element.className = "folder-browser-widget";
        this.element.style.cssText = `
            display: flex;
            gap: 5px;
            align-items: center;
            margin: 2px 0;
            width: 100%;
        `;
        
        // Create path input
        this.pathInput = document.createElement("input");
        this.pathInput.type = "text";
        this.pathInput.value = this.value;
        this.pathInput.placeholder = "Path to video directory";
        this.pathInput.style.cssText = `
            flex: 1;
            padding: 4px 8px;
            border: 1px solid #555;
            background: #333;
            color: #fff;
            border-radius: 3px;
            font-family: monospace;
            font-size: 12px;
        `;
        
        // Create browse button
        this.browseButton = document.createElement("button");
        this.browseButton.textContent = "📁 Browse";
        this.browseButton.title = "Open folder selection dialog";
        this.browseButton.style.cssText = `
            padding: 4px 12px;
            border: 1px solid #666;
            background: #4a4a4a;
            color: #fff;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            white-space: nowrap;
        `;
        
        // Create recent folders dropdown (placeholder)
        this.recentButton = document.createElement("button");
        this.recentButton.textContent = "⏷";
        this.recentButton.title = "Recent folders";
        this.recentButton.style.cssText = `
            padding: 4px 8px;
            border: 1px solid #666;
            background: #4a4a4a;
            color: #fff;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        
        // Assemble widget
        this.element.appendChild(this.pathInput);
        this.element.appendChild(this.browseButton);
        this.element.appendChild(this.recentButton);
        
        // Event handlers
        this.pathInput.addEventListener('input', (e) => {
            this.value = e.target.value;
            this.node.setDirtyCanvas(true, true);
        });
        
        this.browseButton.addEventListener('click', () => {
            this.openFolderDialog();
        });
        
        this.recentButton.addEventListener('click', () => {
            this.showRecentFolders();
        });
    }
    
    async openFolderDialog() {
        try {
            // Try API endpoint first
            const response = await api.fetchApi("/loopycomfy/browse_folder", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    node_id: this.node.id,
                    current_path: this.value
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.success && data.path) {
                    this.value = data.path;
                    this.pathInput.value = data.path;
                    this.node.setDirtyCanvas(true, true);
                    
                    // Store in recent folders
                    this.addToRecent(data.path);
                    
                    // Show success message with video count if available
                    if (data.video_count !== undefined) {
                        console.log(`LoopyComfy: Selected folder with ${data.video_count} videos`);
                        this.showTempMessage(`Selected: ${data.video_count} videos found`);
                    }
                    return;
                } else if (data.fallback_suggestions) {
                    // Show fallback suggestions
                    this.showFallbackSuggestions(data.fallback_suggestions);
                    return;
                }
            }
            
            // Fallback to manual input if API fails
            console.warn("Folder dialog API not available, using manual input");
            this.promptManualInput();
            
        } catch (error) {
            console.warn("Folder dialog API error:", error);
            // Graceful fallback to manual input
            this.promptManualInput();
        }
    }
    
    showTempMessage(message) {
        // Create temporary success indicator
        const indicator = document.createElement("div");
        indicator.textContent = message;
        indicator.style.cssText = `
            position: absolute;
            background: #4a9eff;
            color: white;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 11px;
            z-index: 1000;
            animation: fadeInOut 3s ease;
        `;
        
        // Position near the widget
        this.element.appendChild(indicator);
        
        // Remove after animation
        setTimeout(() => {
            if (indicator.parentNode) {
                indicator.parentNode.removeChild(indicator);
            }
        }, 3000);
    }
    
    promptManualInput() {
        const newPath = prompt("Enter video folder path:\n(API folder browser not available)", this.value);
        if (newPath !== null && newPath.trim() !== "") {
            this.value = newPath;
            this.pathInput.value = newPath;
            this.node.setDirtyCanvas(true, true);
            
            // Store in recent folders
            this.addToRecent(newPath);
        }
    }
    
    showFallbackSuggestions(suggestions) {
        const suggestionText = "Folder browser not available. Try these common paths:\n" +
            suggestions.map((path, i) => `${i+1}. ${path}`).join('\n') +
            "\nEnter number (1-" + suggestions.length + ") or type custom path:";
        
        const selection = prompt(suggestionText);
        
        if (selection !== null) {
            const index = parseInt(selection) - 1;
            if (!isNaN(index) && index >= 0 && index < suggestions.length) {
                // User selected a suggestion
                this.value = suggestions[index];
                this.pathInput.value = suggestions[index];
            } else {
                // User typed custom path
                this.value = selection;
                this.pathInput.value = selection;
            }
            
            this.node.setDirtyCanvas(true, true);
            this.addToRecent(this.value);
        }
    }
    
    showRecentFolders() {
        const recent = this.getRecentFolders();
        if (recent.length === 0) {
            alert("No recent folders");
            return;
        }
        
        // Create simple menu (could be enhanced with proper dropdown)
        const selection = prompt(
            "Recent folders:\n" + recent.map((path, i) => `${i+1}. ${path}`).join('\n') + 
            "\nEnter number (1-" + recent.length + "):"
        );
        
        const index = parseInt(selection) - 1;
        if (!isNaN(index) && index >= 0 && index < recent.length) {
            this.value = recent[index];
            this.pathInput.value = recent[index];
            this.node.setDirtyCanvas(true, true);
        }
    }
    
    getRecentFolders() {
        try {
            const stored = localStorage.getItem('loopycomfy_recent_folders');
            return stored ? JSON.parse(stored) : [];
        } catch {
            return [];
        }
    }
    
    addToRecent(path) {
        try {
            const recent = this.getRecentFolders();
            const filtered = recent.filter(p => p !== path);
            filtered.unshift(path);
            const limited = filtered.slice(0, 5); // Keep only 5 recent
            localStorage.setItem('loopycomfy_recent_folders', JSON.stringify(limited));
        } catch (error) {
            console.warn("Could not save recent folder:", error);
        }
    }
    
    getValue() {
        return this.value;
    }
    
    setValue(value) {
        this.value = value;
        this.pathInput.value = value;
    }
}

// Resolution preset widget with visual previews
class ResolutionPresetWidget {
    constructor(node, inputName, inputData) {
        this.node = node;
        this.inputName = inputName;
        this.presets = inputData[0] || [];
        this.value = inputData[1]?.default || this.presets[0];
        
        // Create main container
        this.element = document.createElement("div");
        this.element.className = "resolution-preset-widget";
        this.element.style.cssText = `
            margin: 2px 0;
            width: 100%;
        `;
        
        // Create dropdown
        this.select = document.createElement("select");
        this.select.style.cssText = `
            width: 100%;
            padding: 4px 8px;
            border: 1px solid #555;
            background: #333;
            color: #fff;
            border-radius: 3px;
            font-family: inherit;
            font-size: 12px;
        `;
        
        // Populate options with aspect ratio indicators
        this.presets.forEach(preset => {
            const option = document.createElement("option");
            option.value = preset;
            option.textContent = preset;
            
            // Add visual indicator for aspect ratio
            if (preset.includes('1920×1080') || preset.includes('1280×720')) {
                option.textContent += " 📺"; // Landscape
            } else if (preset.includes('1080×1920') || preset.includes('9:16')) {
                option.textContent += " 📱"; // Portrait/Mobile
            } else if (preset.includes('1080×1080') || preset.includes('1:1')) {
                option.textContent += " ⬜"; // Square
            } else if (preset.includes('Cinema') || preset.includes('2.35:1')) {
                option.textContent += " 🎬"; // Cinema
            }
            
            this.select.appendChild(option);
        });
        
        this.select.value = this.value;
        
        // Create info display
        this.infoDiv = document.createElement("div");
        this.infoDiv.style.cssText = `
            font-size: 10px;
            color: #aaa;
            margin-top: 2px;
            font-family: monospace;
        `;
        
        // Assemble widget
        this.element.appendChild(this.select);
        this.element.appendChild(this.infoDiv);
        
        // Event handlers
        this.select.addEventListener('change', (e) => {
            this.value = e.target.value;
            this.updateInfo();
            this.node.setDirtyCanvas(true, true);
        });
        
        // Initialize info display
        this.updateInfo();
    }
    
    updateInfo() {
        const preset = this.value;
        
        // Extract dimensions from preset name
        const match = preset.match(/(\d+)×(\d+)/);
        if (match) {
            const width = parseInt(match[1]);
            const height = parseInt(match[2]);
            const aspectRatio = (width / height).toFixed(2);
            const megapixels = ((width * height) / 1000000).toFixed(1);
            
            this.infoDiv.textContent = `${width}×${height} • ${aspectRatio}:1 • ${megapixels}MP`;
            
            // Add orientation indicator
            if (width > height) {
                this.infoDiv.textContent += " • Landscape";
            } else if (height > width) {
                this.infoDiv.textContent += " • Portrait";
            } else {
                this.infoDiv.textContent += " • Square";
            }
        } else if (preset === "Custom") {
            this.infoDiv.textContent = "Use custom width/height inputs below";
        } else {
            this.infoDiv.textContent = "Resolution will be determined by preset";
        }
    }
    
    getValue() {
        return this.value;
    }
    
    setValue(value) {
        this.value = value;
        this.select.value = value;
        this.updateInfo();
    }
}

// Progress indicator widget
class ProgressWidget {
    constructor() {
        this.element = document.createElement("div");
        this.element.className = "progress-widget";
        this.element.style.cssText = `
            margin: 5px 0;
            padding: 8px;
            background: #2a2a2a;
            border: 1px solid #555;
            border-radius: 3px;
            font-size: 11px;
            color: #ccc;
            display: none;
        `;
        
        // Progress bar
        this.progressBar = document.createElement("div");
        this.progressBar.style.cssText = `
            width: 100%;
            height: 16px;
            background: #1a1a1a;
            border-radius: 2px;
            overflow: hidden;
            margin: 4px 0;
        `;
        
        this.progressFill = document.createElement("div");
        this.progressFill.style.cssText = `
            height: 100%;
            background: linear-gradient(90deg, #4a9eff, #00d4aa);
            width: 0%;
            transition: width 0.3s ease;
        `;
        
        this.progressBar.appendChild(this.progressFill);
        
        // Status text
        this.statusText = document.createElement("div");
        this.statusText.style.cssText = `
            text-align: center;
            margin: 2px 0;
        `;
        
        // Memory indicator
        this.memoryText = document.createElement("div");
        this.memoryText.style.cssText = `
            font-size: 10px;
            color: #999;
            text-align: right;
        `;
        
        this.element.appendChild(this.statusText);
        this.element.appendChild(this.progressBar);
        this.element.appendChild(this.memoryText);
    }
    
    show() {
        this.element.style.display = "block";
    }
    
    hide() {
        this.element.style.display = "none";
    }
    
    updateProgress(percent, status = "", memoryInfo = "") {
        this.progressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
        this.statusText.textContent = status;
        this.memoryText.textContent = memoryInfo;
    }
}

// Output Directory Browser Widget for VideoSaver
class OutputDirectoryBrowserWidget {
    constructor(node, inputName, inputData) {
        this.node = node;
        this.inputName = inputName;
        this.value = inputData[1]?.default || "./output/";
        
        // Create main container
        this.element = document.createElement("div");
        this.element.className = "output-directory-browser-widget";
        this.element.style.cssText = `
            display: flex;
            gap: 5px;
            align-items: center;
            margin: 2px 0;
            width: 100%;
        `;
        
        // Create path input
        this.pathInput = document.createElement("input");
        this.pathInput.type = "text";
        this.pathInput.value = this.value;
        this.pathInput.placeholder = "Output directory path";
        this.pathInput.style.cssText = `
            flex: 1;
            padding: 4px 8px;
            border: 1px solid #555;
            background: #333;
            color: #fff;
            border-radius: 3px;
            font-family: monospace;
            font-size: 12px;
        `;
        
        // Create browse button
        this.browseButton = document.createElement("button");
        this.browseButton.textContent = "📁 Browse";
        this.browseButton.title = "Open output directory selection dialog";
        this.browseButton.style.cssText = `
            padding: 4px 12px;
            border: 1px solid #666;
            background: #4a4a4a;
            color: #fff;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            white-space: nowrap;
        `;
        
        // Create create directory button
        this.createButton = document.createElement("button");
        this.createButton.textContent = "+";
        this.createButton.title = "Create new output directory";
        this.createButton.style.cssText = `
            padding: 4px 8px;
            border: 1px solid #666;
            background: #4a4a4a;
            color: #fff;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        
        // Assemble widget
        this.element.appendChild(this.pathInput);
        this.element.appendChild(this.browseButton);
        this.element.appendChild(this.createButton);
        
        // Event handlers
        this.pathInput.addEventListener('input', (e) => {
            this.value = e.target.value;
            this.node.setDirtyCanvas(true, true);
        });
        
        this.browseButton.addEventListener('click', () => {
            this.openDirectoryDialog();
        });
        
        this.createButton.addEventListener('click', () => {
            this.createNewDirectory();
        });
    }
    
    async openDirectoryDialog() {
        try {
            // Try API endpoint for output directory selection
            const response = await api.fetchApi("/loopycomfy/browse_output_dir", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    current_path: this.value
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.success && data.path) {
                    this.value = data.path;
                    this.pathInput.value = data.path;
                    this.node.setDirtyCanvas(true, true);
                    
                    // Show success message with writability status
                    if (data.is_writable !== undefined) {
                        const status = data.is_writable ? "writable" : "read-only";
                        console.log(`LoopyComfy: Selected output directory (${status})`);
                        this.showTempMessage(`Selected: ${data.folder_name} (${status})`);
                    }
                    return;
                } else if (data.fallback_suggestions) {
                    // Show fallback suggestions
                    this.showFallbackSuggestions(data.fallback_suggestions);
                    return;
                }
            }
            
            // Fallback to manual input if API fails
            console.warn("Output directory dialog API not available, using manual input");
            this.promptManualInput();
            
        } catch (error) {
            console.warn("Output directory dialog API error:", error);
            this.promptManualInput();
        }
    }
    
    createNewDirectory() {
        const dirName = prompt("Enter name for new output directory:", "LoopyComfy_Output");
        if (dirName && dirName.trim()) {
            // Create path relative to current directory or use absolute
            const newPath = this.value.includes("/") || this.value.includes("\\") 
                ? `${this.value.replace(/[\/\\]+$/, "")}/${dirName.trim()}`
                : `./${dirName.trim()}`;
            
            this.value = newPath;
            this.pathInput.value = newPath;
            this.node.setDirtyCanvas(true, true);
            
            this.showTempMessage(`Created: ${dirName.trim()}`);
        }
    }
    
    showTempMessage(message) {
        // Create temporary success indicator
        const indicator = document.createElement("div");
        indicator.textContent = message;
        indicator.style.cssText = `
            position: absolute;
            background: #4a9eff;
            color: white;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 11px;
            z-index: 1000;
            animation: fadeInOut 3s ease;
        `;
        
        // Position near the widget
        this.element.appendChild(indicator);
        
        // Remove after animation
        setTimeout(() => {
            if (indicator.parentNode) {
                indicator.parentNode.removeChild(indicator);
            }
        }, 3000);
    }
    
    promptManualInput() {
        const newPath = prompt("Enter output directory path:\n(API directory browser not available)", this.value);
        if (newPath !== null && newPath.trim() !== "") {
            this.value = newPath;
            this.pathInput.value = newPath;
            this.node.setDirtyCanvas(true, true);
        }
    }
    
    showFallbackSuggestions(suggestions) {
        const suggestionText = "Directory browser not available. Try these common paths:\n" +
            suggestions.map((path, i) => `${i+1}. ${path}`).join('\n') +
            "\nEnter number (1-" + suggestions.length + ") or type custom path:";
        
        const selection = prompt(suggestionText);
        
        if (selection !== null) {
            const index = parseInt(selection) - 1;
            if (!isNaN(index) && index >= 0 && index < suggestions.length) {
                // User selected a suggestion
                this.value = suggestions[index];
                this.pathInput.value = suggestions[index];
            } else {
                // User typed custom path
                this.value = selection;
                this.pathInput.value = selection;
            }
            
            this.node.setDirtyCanvas(true, true);
        }
    }
    
    getValue() {
        return this.value;
    }
    
    setValue(value) {
        this.value = value;
        this.pathInput.value = value;
    }
}

// Register UI extensions
app.registerExtension({
    name: "LoopyComfy.NonLinearAvatarUI",
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // Enhanced VideoAssetLoader with folder browser
        if (nodeData.name === "LoopyComfy_VideoAssetLoader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Add folder browser widget
                if (this.widgets) {
                    const pathWidget = this.widgets.find(w => w.name === "directory_path");
                    if (pathWidget) {
                        // Replace with custom folder browser widget
                        const folderBrowser = new FolderBrowserWidget(this, "directory_path", pathWidget.options);
                        
                        // Hide original widget and add custom one
                        pathWidget.computeSize = () => [0, -4]; // Hide original
                        
                        // Add custom widget element to node
                        if (!this.folderBrowserWidget) {
                            this.folderBrowserWidget = folderBrowser;
                            
                            // Add to DOM when node is being rendered
                            const originalOnDrawForeground = this.onDrawForeground;
                            this.onDrawForeground = function(ctx) {
                                const result = originalOnDrawForeground?.apply(this, arguments);
                                
                                // Ensure our widget is in the DOM
                                if (this.folderBrowserWidget && this.folderBrowserWidget.element.parentNode !== document.body) {
                                    // For now, we'll work within ComfyUI's widget system constraints
                                    // The actual folder dialog will be triggered via API calls
                                }
                                
                                return result;
                            };
                        }
                    }
                }
                
                return result;
            };
        }
        
        // Enhanced VideoSaver with output directory browser
        if (nodeData.name === "LoopyComfy_VideoSaver") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Add output directory browser widget
                if (this.widgets) {
                    const outputDirWidget = this.widgets.find(w => w.name === "output_directory");
                    if (outputDirWidget) {
                        // Replace with custom output directory browser widget
                        const directoryBrowser = new OutputDirectoryBrowserWidget(this, "output_directory", outputDirWidget.options);
                        
                        // Hide original widget and add custom one
                        outputDirWidget.computeSize = () => [0, -4]; // Hide original
                        
                        // Add custom widget element to node
                        if (!this.outputDirectoryBrowserWidget) {
                            this.outputDirectoryBrowserWidget = directoryBrowser;
                        }
                    }
                }
                
                return result;
            };
        }
        
        // Enhanced VideoSequenceComposer with resolution previews
        if (nodeData.name === "LoopyComfy_VideoSequenceComposer") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Add progress widget
                this.progressWidget = new ProgressWidget();
                
                // Monitor execution progress
                const originalOnExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    if (this.progressWidget) {
                        this.progressWidget.hide();
                    }
                    return originalOnExecuted?.apply(this, arguments);
                };
                
                return result;
            };
        }
    },
    
    async setup() {
        // Add CSS for our custom widgets
        const style = document.createElement("style");
        style.textContent = `
            .folder-browser-widget button:hover {
                background: #5a5a5a !important;
                border-color: #777 !important;
            }
            
            .resolution-preset-widget select:focus {
                outline: none;
                border-color: #4a9eff;
                box-shadow: 0 0 3px rgba(74, 158, 255, 0.3);
            }
            
            .progress-widget {
                animation: slideDown 0.3s ease-out;
            }
            
            @keyframes slideDown {
                from {
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes fadeInOut {
                0% {
                    opacity: 0;
                    transform: translateY(-5px);
                }
                20% {
                    opacity: 1;
                    transform: translateY(0);
                }
                80% {
                    opacity: 1;
                    transform: translateY(0);
                }
                100% {
                    opacity: 0;
                    transform: translateY(5px);
                }
            }
            
            .output-directory-browser-widget button:hover {
                background: #5a5a5a !important;
                border-color: #777 !important;
            }
            
            /* Tooltip enhancements */
            .comfy-tooltip {
                max-width: 300px;
                font-size: 12px;
                line-height: 1.4;
                padding: 8px 12px;
                background: rgba(0, 0, 0, 0.9);
                border: 1px solid #666;
                border-radius: 4px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            }
        `;
        document.head.appendChild(style);
        
        console.log("LoopyComfy Non-Linear Avatar UI extensions loaded");
    }
});

// API endpoint for folder browsing (to be handled by the Python backend)
api.addEventListener("status", (event) => {
    const status = event.detail;
    
    // Update progress widgets for any active LoopyComfy nodes
    document.querySelectorAll('.progress-widget').forEach(widget => {
        if (status.exec_info && status.exec_info.node_type?.includes('LoopyComfy')) {
            widget.style.display = 'block';
            // Update progress based on status
        }
    });
});

export { FolderBrowserWidget, OutputDirectoryBrowserWidget, ResolutionPresetWidget, ProgressWidget };