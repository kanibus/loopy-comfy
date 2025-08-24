/**
 * Real-Time Preview Client for LoopyComfy v2.0
 * 
 * Provides real-time video preview with browser fallback support,
 * automatic reconnection, and adaptive quality.
 */

class RealTimePreview {
    constructor(options = {}) {
        this.token = options.token || '';
        this.serverUrl = options.serverUrl || 'ws://localhost:8765';
        this.canvasId = options.canvasId || 'preview-canvas';
        this.containerId = options.containerId || 'preview-container';
        
        // UI Elements
        this.canvas = document.getElementById(this.canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.container = document.getElementById(this.containerId);
        
        // Connection state
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectTimeout = null;
        
        // Decoder state
        this.decoder = null;
        this.decoderType = 'canvas';
        this.capabilities = {};
        
        // Performance tracking
        this.stats = {
            framesReceived: 0,
            bytesReceived: 0,
            lastFrameTime: 0,
            fps: 0,
            latency: 0
        };
        
        // Quality settings
        this.quality = 'medium';
        this.preferredEncoding = 'jpeg';
        
        this.init();
    }
    
    init() {
        this.detectCapabilities();
        this.setupUI();
        this.connect();
    }
    
    detectCapabilities() {
        this.capabilities = {
            // Check VideoDecoder API (Chrome 94+, Edge 94+)
            videoDecoder: typeof VideoDecoder !== 'undefined',
            
            // Check WebCodecs support
            webCodecs: typeof VideoDecoder !== 'undefined' && 
                      typeof VideoEncoder !== 'undefined',
            
            // Check Media Source Extensions
            mediaSource: typeof MediaSource !== 'undefined',
            
            // Check WebGL support
            webgl: this.checkWebGLSupport(),
            
            // Check WebP support
            webp: this.checkWebPSupport(),
            
            // Check for hardware acceleration hints
            hardwareAcceleration: this.checkHardwareAcceleration()
        };
        
        // Determine best decoder
        if (this.capabilities.videoDecoder) {
            this.decoderType = 'videoDecoder';
            this.preferredEncoding = 'h264';
        } else if (this.capabilities.mediaSource) {
            this.decoderType = 'mediaSource';
            this.preferredEncoding = this.capabilities.webp ? 'webp' : 'jpeg';
        } else {
            this.decoderType = 'canvas';
            this.preferredEncoding = this.capabilities.webp ? 'webp' : 'jpeg';
        }
        
        console.log('Detected capabilities:', this.capabilities);
        console.log('Using decoder:', this.decoderType);
        console.log('Preferred encoding:', this.preferredEncoding);
        
        this.initDecoder();
    }
    
    checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            return !!gl;
        } catch (e) {
            return false;
        }
    }
    
    checkWebPSupport() {
        return new Promise((resolve) => {
            const webp = new Image();
            webp.onload = webp.onerror = () => resolve(webp.height === 2);
            webp.src = 'data:image/webp;base64,UklGRjoAAABXRUJQVlA4IC4AAACyAgCdASoCAAIALmk0mk0iIiIiIgBoSygABc6WWgAA/veff/0PP8bA//LwYAAA';
        });
    }
    
    checkHardwareAcceleration() {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl');
            if (!gl) return false;
            
            const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
            if (debugInfo) {
                const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
                return renderer.toLowerCase().includes('nvidia') || 
                       renderer.toLowerCase().includes('amd') ||
                       renderer.toLowerCase().includes('intel');
            }
            return false;
        } catch (e) {
            return false;
        }
    }
    
    initDecoder() {
        switch (this.decoderType) {
            case 'videoDecoder':
                this.initVideoDecoder();
                break;
            case 'mediaSource':
                this.initMediaSourceDecoder();
                break;
            default:
                this.initCanvasDecoder();
        }
    }
    
    initVideoDecoder() {
        try {
            this.decoder = new VideoDecoder({
                output: (frame) => {
                    this.renderFrame(frame);
                    frame.close();
                },
                error: (e) => {
                    console.error('VideoDecoder error:', e);
                    this.fallbackToCanvas();
                }
            });
            
            // Configure for H.264
            this.decoder.configure({
                codec: 'avc1.42E01E', // H.264 Baseline Profile Level 3.0
                optimizeForLatency: true
            });
            
        } catch (e) {
            console.warn('VideoDecoder initialization failed:', e);
            this.fallbackToCanvas();
        }
    }
    
    initMediaSourceDecoder() {
        // Media Source Extensions fallback
        this.decoderType = 'canvas'; // Simplify for now
        this.initCanvasDecoder();
    }
    
    initCanvasDecoder() {
        this.decoder = {
            decode: (data, encoding) => {
                const img = new Image();
                img.onload = () => {
                    if (this.ctx && this.canvas) {
                        this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
                    }
                };
                img.onerror = (e) => {
                    console.error('Image decode error:', e);
                };
                
                const mimeType = encoding === 'webp' ? 'image/webp' : 'image/jpeg';
                img.src = `data:${mimeType};base64,${data}`;
            }
        };
    }
    
    setupUI() {
        if (!this.container) {
            console.warn('Preview container not found');
            return;
        }
        
        // Add control panel
        const controlPanel = document.createElement('div');
        controlPanel.className = 'preview-controls';
        controlPanel.innerHTML = `
            <div class="connection-status">
                <span id="connection-indicator" class="status-disconnected">●</span>
                <span id="connection-text">Disconnected</span>
            </div>
            <div class="quality-controls">
                <label for="quality-select">Quality:</label>
                <select id="quality-select">
                    <option value="low">Low</option>
                    <option value="medium" selected>Medium</option>
                    <option value="high">High</option>
                </select>
            </div>
            <div class="stats-display">
                <span id="fps-display">FPS: --</span>
                <span id="latency-display">Latency: --</span>
            </div>
        `;
        
        this.container.appendChild(controlPanel);
        
        // Setup event listeners
        document.getElementById('quality-select').addEventListener('change', (e) => {
            this.changeQuality(e.target.value);
        });
        
        // Add CSS styles
        this.addStyles();
    }
    
    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .preview-controls {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                background: #f0f0f0;
                border-radius: 5px;
                margin-bottom: 10px;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }
            
            .connection-status {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            .status-connected { color: #4CAF50; }
            .status-disconnected { color: #f44336; }
            .status-connecting { color: #ff9800; }
            
            .quality-controls select {
                padding: 4px 8px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            
            .stats-display {
                display: flex;
                gap: 15px;
                font-family: monospace;
            }
            
            #preview-canvas {
                max-width: 100%;
                height: auto;
                border: 1px solid #ccc;
            }
        `;
        document.head.appendChild(style);
    }
    
    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || 
                        this.ws.readyState === WebSocket.OPEN)) {
            return;
        }
        
        this.updateConnectionStatus('connecting', 'Connecting...');
        
        const url = `${this.serverUrl}?token=${encodeURIComponent(this.token)}`;
        
        try {
            this.ws = new WebSocket(url);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('connected', 'Connected');
                
                // Send capabilities
                this.sendCapabilities();
            };
            
            this.ws.onmessage = (event) => {
                this.handleMessage(event.data);
            };
            
            this.ws.onclose = (event) => {
                console.log('WebSocket disconnected:', event.code, event.reason);
                this.isConnected = false;
                this.updateConnectionStatus('disconnected', 'Disconnected');
                this.attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('disconnected', 'Connection Error');
            };
            
        } catch (e) {
            console.error('WebSocket connection failed:', e);
            this.updateConnectionStatus('disconnected', 'Connection Failed');
            this.attemptReconnect();
        }
    }
    
    sendCapabilities() {
        const message = {
            type: 'capabilities',
            capabilities: {
                supports_h264: this.capabilities.videoDecoder,
                supports_webp: this.capabilities.webp,
                supports_webgl: this.capabilities.webgl,
                hardware_acceleration: this.capabilities.hardwareAcceleration,
                preferred_encoding: this.preferredEncoding,
                max_resolution: [1920, 1080],
                canvas_size: this.canvas ? [this.canvas.width, this.canvas.height] : [1920, 1080]
            }
        };
        
        this.send(message);
    }
    
    handleMessage(data) {
        try {
            const message = JSON.parse(data);
            
            switch (message.type) {
                case 'capabilities':
                    this.handleCapabilities(message);
                    break;
                    
                case 'frame':
                    this.handleFrame(message);
                    break;
                    
                case 'notification':
                    this.handleNotification(message);
                    break;
                    
                case 'stats':
                    this.handleStats(message.data);
                    break;
                    
                case 'error':
                    console.error('Server error:', message.message);
                    this.showError(message.message);
                    break;
                    
                case 'pong':
                    this.handlePong(message);
                    break;
            }
            
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    }
    
    handleCapabilities(message) {
        console.log('Server capabilities:', message);
        
        // Adjust preferred encoding based on server support
        const serverEncodings = message.encodings || {};
        
        if (this.preferredEncoding === 'h264' && !serverEncodings.h264) {
            this.preferredEncoding = serverEncodings.webp ? 'webp' : 'jpeg';
        }
        
        if (this.preferredEncoding === 'webp' && !serverEncodings.webp) {
            this.preferredEncoding = 'jpeg';
        }
        
        console.log('Adjusted preferred encoding:', this.preferredEncoding);
    }
    
    handleFrame(message) {
        const frameTime = Date.now();
        
        try {
            // Update stats
            this.stats.framesReceived++;
            this.stats.bytesReceived += message.data.length;
            this.stats.latency = frameTime - (message.timestamp * 1000);
            
            if (this.stats.lastFrameTime > 0) {
                const timeDelta = frameTime - this.stats.lastFrameTime;
                this.stats.fps = 1000 / timeDelta;
            }
            this.stats.lastFrameTime = frameTime;
            
            // Decode and render frame
            if (this.decoderType === 'videoDecoder' && message.encoding === 'h264') {
                this.decodeH264Frame(message);
            } else {
                // Use canvas decoder for JPEG/WebP
                this.decoder.decode(message.data, message.encoding);
            }
            
            // Update UI stats
            this.updateStatsDisplay();
            
        } catch (e) {
            console.error('Frame handling error:', e);
            this.fallbackToCanvas();
        }
    }
    
    decodeH264Frame(message) {
        try {
            const frameData = Uint8Array.from(atob(message.data), c => c.charCodeAt(0));
            
            const chunk = new EncodedVideoChunk({
                type: 'key', // Assume all frames are keyframes for simplicity
                timestamp: message.timestamp * 1000000, // Convert to microseconds
                data: frameData
            });
            
            this.decoder.decode(chunk);
            
        } catch (e) {
            console.error('H.264 decode error:', e);
            this.fallbackToCanvas();
        }
    }
    
    renderFrame(videoFrame) {
        if (!this.canvas || !this.ctx) return;
        
        try {
            // Draw VideoFrame to canvas
            this.ctx.drawImage(videoFrame, 0, 0, this.canvas.width, this.canvas.height);
        } catch (e) {
            console.error('Frame render error:', e);
        }
    }
    
    handleNotification(message) {
        console.log('Server notification:', message.message);
        this.showNotification(message.message, message.notification_type);
    }
    
    handleStats(stats) {
        console.log('Server stats:', stats);
        // Could update UI with server-side statistics
    }
    
    handlePong(message) {
        // Calculate round-trip time
        const rtt = Date.now() - this.lastPingTime;
        this.stats.latency = rtt;
    }
    
    fallbackToCanvas() {
        console.log('Falling back to canvas rendering');
        this.decoderType = 'canvas';
        this.preferredEncoding = this.capabilities.webp ? 'webp' : 'jpeg';
        this.initCanvasDecoder();
        
        // Notify server of capability change
        this.sendCapabilities();
    }
    
    changeQuality(quality) {
        this.quality = quality;
        this.send({
            type: 'quality_request',
            quality: quality
        });
    }
    
    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.warn('Cannot send message: WebSocket not connected');
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.updateConnectionStatus('disconnected', 'Connection Failed');
            return;
        }
        
        this.reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 10000);
        
        console.log(`Reconnecting in ${delay}ms... (attempt ${this.reconnectAttempts})`);
        this.updateConnectionStatus('connecting', `Reconnecting... (${this.reconnectAttempts})`);
        
        this.reconnectTimeout = setTimeout(() => {
            this.connect();
        }, delay);
    }
    
    updateConnectionStatus(status, text) {
        const indicator = document.getElementById('connection-indicator');
        const statusText = document.getElementById('connection-text');
        
        if (indicator && statusText) {
            indicator.className = `status-${status}`;
            statusText.textContent = text;
        }
    }
    
    updateStatsDisplay() {
        const fpsDisplay = document.getElementById('fps-display');
        const latencyDisplay = document.getElementById('latency-display');
        
        if (fpsDisplay) {
            fpsDisplay.textContent = `FPS: ${this.stats.fps.toFixed(1)}`;
        }
        
        if (latencyDisplay) {
            latencyDisplay.textContent = `Latency: ${this.stats.latency.toFixed(0)}ms`;
        }
    }
    
    showError(message) {
        console.error('Preview error:', message);
        // Could show toast notification or error overlay
    }
    
    showNotification(message, type = 'info') {
        console.log(`Notification (${type}):`, message);
        // Could show toast notification
    }
    
    startPing() {
        setInterval(() => {
            if (this.isConnected) {
                this.lastPingTime = Date.now();
                this.send({ type: 'ping', timestamp: this.lastPingTime });
            }
        }, 5000); // Ping every 5 seconds
    }
    
    disconnect() {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }
        
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        this.isConnected = false;
        this.updateConnectionStatus('disconnected', 'Disconnected');
    }
    
    getStats() {
        return { ...this.stats };
    }
    
    getCapabilities() {
        return { ...this.capabilities };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RealTimePreview;
} else if (typeof window !== 'undefined') {
    window.RealTimePreview = RealTimePreview;
}