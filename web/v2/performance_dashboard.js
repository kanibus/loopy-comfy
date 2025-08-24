/**
 * Performance Dashboard for LoopyComfy v2.0
 * 
 * Provides real-time performance monitoring with graceful degradation
 * for unsupported browsers and comprehensive error handling.
 */

class PerformanceDashboard {
    constructor(options = {}) {
        this.containerId = options.containerId || 'performance-dashboard';
        this.updateInterval = options.updateInterval || 1000; // 1 second
        this.historyLength = options.historyLength || 60; // 60 seconds
        this.serverUrl = options.serverUrl || 'http://localhost:8188';
        
        // Metrics storage
        this.metrics = {
            fps: 0,
            gpuUsage: 0,
            gpuMemoryUsage: 0,
            memoryUsage: 0,
            processingTime: 0,
            queueLength: 0,
            mlInferenceTime: 0,
            fallbackCount: 0,
            errorCount: 0,
            timestamp: Date.now()
        };
        
        // Historical data for charts
        this.history = {
            fps: [],
            memory: [],
            gpu: [],
            processing: [],
            ml: []
        };
        
        // Chart instances
        this.charts = {};
        this.displayMode = 'auto'; // auto, charts, text
        
        // Connection state
        this.isConnected = false;
        this.consecutiveErrors = 0;
        this.updateTimer = null;
        
        this.init();
    }
    
    init() {
        this.detectDisplayMode();
        this.setupContainer();
        this.initializeDisplay();
        this.startMonitoring();
    }
    
    detectDisplayMode() {
        // Check for charting libraries
        if (typeof Chart !== 'undefined') {
            this.displayMode = 'charts';
            this.chartLibrary = 'chartjs';
        } else if (typeof SmoothieChart !== 'undefined') {
            this.displayMode = 'charts';
            this.chartLibrary = 'smoothie';
        } else if (typeof d3 !== 'undefined') {
            this.displayMode = 'charts';
            this.chartLibrary = 'd3';
        } else {
            this.displayMode = 'text';
            this.chartLibrary = null;
        }
        
        console.log(`Performance dashboard mode: ${this.displayMode} (${this.chartLibrary || 'none'})`);
    }
    
    setupContainer() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`Performance dashboard container not found: ${this.containerId}`);
            return;
        }
        
        container.className = 'performance-dashboard';
        
        // Add CSS if not present
        if (!document.querySelector('#performance-dashboard-styles')) {
            this.addStyles();
        }
    }
    
    addStyles() {
        const style = document.createElement('style');
        style.id = 'performance-dashboard-styles';
        style.textContent = `
            .performance-dashboard {
                font-family: 'Monaco', 'Consolas', monospace;
                background: #1a1a1a;
                color: #e0e0e0;
                border-radius: 8px;
                padding: 16px;
                margin: 8px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            }
            
            .dashboard-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 16px;
                border-bottom: 1px solid #333;
                padding-bottom: 8px;
            }
            
            .dashboard-title {
                font-size: 18px;
                font-weight: bold;
                color: #4CAF50;
            }
            
            .connection-status {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .status-indicator {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            
            .status-connected { background-color: #4CAF50; }
            .status-disconnected { background-color: #f44336; }
            .status-warning { background-color: #ff9800; }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-bottom: 16px;
            }
            
            .metric-card {
                background: #2a2a2a;
                border-radius: 6px;
                padding: 12px;
                border-left: 4px solid #4CAF50;
            }
            
            .metric-card.warning {
                border-left-color: #ff9800;
            }
            
            .metric-card.error {
                border-left-color: #f44336;
            }
            
            .metric-label {
                font-size: 12px;
                color: #888;
                margin-bottom: 4px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .metric-value {
                font-size: 20px;
                font-weight: bold;
                color: #e0e0e0;
            }
            
            .metric-unit {
                font-size: 14px;
                color: #888;
                margin-left: 4px;
            }
            
            .metric-trend {
                font-size: 11px;
                margin-top: 4px;
            }
            
            .trend-up { color: #4CAF50; }
            .trend-down { color: #f44336; }
            .trend-stable { color: #888; }
            
            .charts-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 16px;
            }
            
            .chart-card {
                background: #2a2a2a;
                border-radius: 6px;
                padding: 16px;
            }
            
            .chart-title {
                font-size: 14px;
                color: #4CAF50;
                margin-bottom: 12px;
                text-align: center;
            }
            
            .chart-canvas {
                width: 100%;
                height: 200px;
            }
            
            .text-display {
                font-family: monospace;
                background: #2a2a2a;
                padding: 16px;
                border-radius: 6px;
                white-space: pre-line;
                line-height: 1.6;
            }
            
            .error-message {
                background: #1a0000;
                border: 1px solid #f44336;
                color: #ffcdd2;
                padding: 12px;
                border-radius: 4px;
                margin-top: 8px;
            }
            
            .loading-spinner {
                display: inline-block;
                width: 16px;
                height: 16px;
                border: 2px solid #333;
                border-radius: 50%;
                border-top-color: #4CAF50;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
    }
    
    initializeDisplay() {
        const container = document.getElementById(this.containerId);
        if (!container) return;
        
        // Clear container
        container.innerHTML = '';
        
        // Add header
        const header = document.createElement('div');
        header.className = 'dashboard-header';
        header.innerHTML = `
            <div class="dashboard-title">Performance Monitor</div>
            <div class="connection-status">
                <div class="status-indicator status-disconnected" id="connection-indicator"></div>
                <span id="connection-text">Initializing...</span>
                <div class="loading-spinner"></div>
            </div>
        `;
        container.appendChild(header);
        
        // Add metrics display based on mode
        if (this.displayMode === 'charts') {
            this.initializeCharts(container);
        } else {
            this.initializeTextDisplay(container);
        }
    }
    
    initializeCharts(container) {
        // Metrics cards
        const metricsGrid = document.createElement('div');
        metricsGrid.className = 'metrics-grid';
        
        const metricCards = [
            { id: 'fps', label: 'FPS', unit: '' },
            { id: 'memory', label: 'Memory', unit: 'GB' },
            { id: 'gpu', label: 'GPU', unit: '%' },
            { id: 'processing', label: 'Processing', unit: 'ms' },
            { id: 'ml', label: 'ML Inference', unit: 'ms' },
            { id: 'queue', label: 'Queue', unit: '' }
        ];
        
        metricCards.forEach(metric => {
            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = `
                <div class="metric-label">${metric.label}</div>
                <div class="metric-value" id="${metric.id}-value">--<span class="metric-unit">${metric.unit}</span></div>
                <div class="metric-trend" id="${metric.id}-trend">--</div>
            `;
            metricsGrid.appendChild(card);
        });
        
        container.appendChild(metricsGrid);
        
        // Charts container
        const chartsContainer = document.createElement('div');
        chartsContainer.className = 'charts-container';
        
        const charts = [
            { id: 'fps-chart', title: 'FPS Over Time' },
            { id: 'memory-chart', title: 'Memory Usage' },
            { id: 'processing-chart', title: 'Processing Time' }
        ];
        
        charts.forEach(chart => {
            const chartCard = document.createElement('div');
            chartCard.className = 'chart-card';
            chartCard.innerHTML = `
                <div class="chart-title">${chart.title}</div>
                <canvas id="${chart.id}" class="chart-canvas"></canvas>
            `;
            chartsContainer.appendChild(chartCard);
        });
        
        container.appendChild(chartsContainer);
        
        // Initialize chart library
        this.createCharts();
    }
    
    initializeTextDisplay(container) {
        const textDisplay = document.createElement('div');
        textDisplay.className = 'text-display';
        textDisplay.id = 'text-metrics';
        textDisplay.textContent = 'Loading metrics...';
        container.appendChild(textDisplay);
    }
    
    createCharts() {
        if (this.chartLibrary === 'chartjs') {
            this.createChartJSCharts();
        } else if (this.chartLibrary === 'smoothie') {
            this.createSmoothieCharts();
        }
    }
    
    createChartJSCharts() {
        const defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'second',
                        displayFormats: { second: 'HH:mm:ss' }
                    },
                    ticks: { color: '#888' }
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: '#888' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#e0e0e0' }
                }
            },
            elements: {
                line: { tension: 0.2 },
                point: { radius: 2 }
            }
        };
        
        // FPS Chart
        const fpsCtx = document.getElementById('fps-chart');
        if (fpsCtx) {
            this.charts.fps = new Chart(fpsCtx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'FPS',
                        data: [],
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        fill: true
                    }]
                },
                options: { ...defaultOptions, scales: { ...defaultOptions.scales, y: { ...defaultOptions.scales.y, max: 60 }}}
            });
        }
        
        // Memory Chart
        const memoryCtx = document.getElementById('memory-chart');
        if (memoryCtx) {
            this.charts.memory = new Chart(memoryCtx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'System Memory',
                        data: [],
                        borderColor: '#2196F3',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        fill: true
                    }, {
                        label: 'GPU Memory',
                        data: [],
                        borderColor: '#FF9800',
                        backgroundColor: 'rgba(255, 152, 0, 0.1)',
                        fill: true
                    }]
                },
                options: { ...defaultOptions, scales: { ...defaultOptions.scales, y: { ...defaultOptions.scales.y, max: 100 }}}
            });
        }
        
        // Processing Chart
        const processingCtx = document.getElementById('processing-chart');
        if (processingCtx) {
            this.charts.processing = new Chart(processingCtx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'Processing Time',
                        data: [],
                        borderColor: '#9C27B0',
                        backgroundColor: 'rgba(156, 39, 176, 0.1)',
                        fill: true
                    }, {
                        label: 'ML Inference',
                        data: [],
                        borderColor: '#F44336',
                        backgroundColor: 'rgba(244, 67, 54, 0.1)',
                        fill: true
                    }]
                },
                options: defaultOptions
            });
        }
    }
    
    async startMonitoring() {
        this.updateTimer = setInterval(() => {
            this.updateMetrics();
        }, this.updateInterval);
        
        // Initial update
        await this.updateMetrics();
    }
    
    async updateMetrics() {
        try {
            const response = await fetch(`${this.serverUrl}/loopycomfy/performance_metrics`, {
                method: 'GET',
                timeout: 5000
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Update metrics
            this.metrics = { ...this.metrics, ...data, timestamp: Date.now() };
            
            // Update history
            this.updateHistory();
            
            // Update displays
            this.updateDisplays();
            
            // Update connection status
            this.updateConnectionStatus('connected');
            this.consecutiveErrors = 0;
            
        } catch (error) {
            console.error('Failed to update metrics:', error);
            this.consecutiveErrors++;
            
            if (this.consecutiveErrors > 5) {
                this.updateConnectionStatus('disconnected', `Connection lost: ${error.message}`);
            } else {
                this.updateConnectionStatus('warning', `Connection issues (${this.consecutiveErrors})`);
            }
            
            // Continue with cached metrics
            this.updateDisplays();
        }
    }
    
    updateHistory() {
        const timestamp = this.metrics.timestamp;
        
        // Add current metrics to history
        this.history.fps.push({ x: timestamp, y: this.metrics.fps });
        this.history.memory.push({ 
            x: timestamp, 
            y: this.metrics.memoryUsage / (1024 * 1024 * 1024) // Convert to GB
        });
        this.history.gpu.push({ x: timestamp, y: this.metrics.gpuUsage });
        this.history.processing.push({ x: timestamp, y: this.metrics.processingTime });
        this.history.ml.push({ x: timestamp, y: this.metrics.mlInferenceTime });
        
        // Trim history to keep only recent data
        const cutoffTime = timestamp - (this.historyLength * 1000);
        Object.keys(this.history).forEach(key => {
            this.history[key] = this.history[key].filter(point => point.x > cutoffTime);
        });
    }
    
    updateDisplays() {
        if (this.displayMode === 'charts') {
            this.updateChartsDisplay();
            this.updateMetricCards();
        } else {
            this.updateTextDisplay();
        }
    }
    
    updateMetricCards() {
        const updates = [
            { id: 'fps', value: this.metrics.fps, unit: '', threshold: [25, 15] },
            { id: 'memory', value: this.metrics.memoryUsage / (1024**3), unit: 'GB', threshold: [12, 16] },
            { id: 'gpu', value: this.metrics.gpuUsage, unit: '%', threshold: [70, 90] },
            { id: 'processing', value: this.metrics.processingTime, unit: 'ms', threshold: [100, 200] },
            { id: 'ml', value: this.metrics.mlInferenceTime, unit: 'ms', threshold: [50, 100] },
            { id: 'queue', value: this.metrics.queueLength, unit: '', threshold: [5, 10] }
        ];
        
        updates.forEach(({ id, value, unit, threshold }) => {
            const valueElement = document.getElementById(`${id}-value`);
            const trendElement = document.getElementById(`${id}-trend`);
            const cardElement = valueElement?.closest('.metric-card');
            
            if (valueElement) {
                const displayValue = typeof value === 'number' ? value.toFixed(value < 10 ? 2 : 1) : '--';
                valueElement.innerHTML = `${displayValue}<span class="metric-unit">${unit}</span>`;
            }
            
            if (trendElement && this.history[id] && this.history[id].length >= 2) {
                const recent = this.history[id].slice(-2);
                const trend = recent[1].y - recent[0].y;
                const trendPercent = Math.abs(trend / recent[0].y * 100);
                
                if (Math.abs(trend) < 0.1) {
                    trendElement.textContent = 'Stable';
                    trendElement.className = 'metric-trend trend-stable';
                } else if (trend > 0) {
                    trendElement.textContent = `+${trendPercent.toFixed(1)}%`;
                    trendElement.className = 'metric-trend trend-up';
                } else {
                    trendElement.textContent = `-${trendPercent.toFixed(1)}%`;
                    trendElement.className = 'metric-trend trend-down';
                }
            }
            
            // Update card status based on thresholds
            if (cardElement && threshold && typeof value === 'number') {
                cardElement.className = 'metric-card';
                if (value > threshold[1]) {
                    cardElement.classList.add('error');
                } else if (value > threshold[0]) {
                    cardElement.classList.add('warning');
                }
            }
        });
    }
    
    updateChartsDisplay() {
        if (this.chartLibrary !== 'chartjs' || !this.charts) return;
        
        // Update FPS chart
        if (this.charts.fps) {
            this.charts.fps.data.datasets[0].data = this.history.fps.slice();
            this.charts.fps.update('none');
        }
        
        // Update Memory chart
        if (this.charts.memory) {
            this.charts.memory.data.datasets[0].data = this.history.memory.slice();
            this.charts.memory.data.datasets[1].data = this.history.gpu.slice();
            this.charts.memory.update('none');
        }
        
        // Update Processing chart
        if (this.charts.processing) {
            this.charts.processing.data.datasets[0].data = this.history.processing.slice();
            this.charts.processing.data.datasets[1].data = this.history.ml.slice();
            this.charts.processing.update('none');
        }
    }
    
    updateTextDisplay() {
        const textElement = document.getElementById('text-metrics');
        if (!textElement) return;
        
        const memoryGB = (this.metrics.memoryUsage / (1024**3)).toFixed(2);
        
        textElement.textContent = `
Performance Metrics (${new Date(this.metrics.timestamp).toLocaleTimeString()})

┌─────────────────────────────────────────────┐
│ Real-time Performance                       │
├─────────────────────────────────────────────┤
│ FPS:              ${this.metrics.fps.toFixed(1).padStart(8)} fps     │
│ Processing Time:  ${this.metrics.processingTime.toFixed(0).padStart(8)} ms      │
│ ML Inference:     ${this.metrics.mlInferenceTime.toFixed(0).padStart(8)} ms      │
├─────────────────────────────────────────────┤
│ System Resources                            │
├─────────────────────────────────────────────┤
│ Memory Usage:     ${memoryGB.padStart(8)} GB      │
│ GPU Usage:        ${this.metrics.gpuUsage.toFixed(0).padStart(8)}%       │
│ Queue Length:     ${this.metrics.queueLength.toString().padStart(8)}         │
├─────────────────────────────────────────────┤
│ Error Tracking                              │
├─────────────────────────────────────────────┤
│ Fallbacks:        ${this.metrics.fallbackCount.toString().padStart(8)}         │
│ Errors:           ${this.metrics.errorCount.toString().padStart(8)}         │
└─────────────────────────────────────────────┘
        `;
    }
    
    updateConnectionStatus(status, message = null) {
        const indicator = document.getElementById('connection-indicator');
        const text = document.getElementById('connection-text');
        const spinner = document.querySelector('.loading-spinner');
        
        if (!indicator || !text) return;
        
        // Update indicator
        indicator.className = `status-indicator status-${status}`;
        
        // Update text
        const statusMessages = {
            connected: 'Connected',
            disconnected: 'Disconnected',
            warning: 'Connection Issues'
        };
        
        text.textContent = message || statusMessages[status] || 'Unknown';
        
        // Update spinner visibility
        if (spinner) {
            spinner.style.display = status === 'connected' ? 'none' : 'inline-block';
        }
        
        this.isConnected = status === 'connected';
    }
    
    getMetrics() {
        return { ...this.metrics };
    }
    
    getHistory() {
        return { ...this.history };
    }
    
    destroy() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
        
        // Destroy charts
        if (this.charts) {
            Object.values(this.charts).forEach(chart => {
                if (chart && typeof chart.destroy === 'function') {
                    chart.destroy();
                }
            });
            this.charts = {};
        }
        
        console.log('Performance dashboard destroyed');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PerformanceDashboard;
} else if (typeof window !== 'undefined') {
    window.PerformanceDashboard = PerformanceDashboard;
}