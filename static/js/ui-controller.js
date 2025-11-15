/**
 * UIController Component
 * Manages UI interactions and coordinates between VideoPlayer and user interface
 */
class UIController {
    /**
     * Initialize UIController with VideoPlayer instance
     * @param {VideoPlayer} videoPlayer - VideoPlayer instance to control
     */
    constructor(videoPlayer) {
        if (!videoPlayer) {
            throw new Error('VideoPlayer instance required');
        }
        
        this.videoPlayer = videoPlayer;
        this.features = [];
        this.currentFeature = null;
        
        // Store DOM element references
        this.featureTogglesContainer = document.getElementById('feature-toggles');
        this.playPauseButton = document.getElementById('play-pause-btn');
        
        if (!this.featureTogglesContainer) {
            console.error('Feature toggles container not found');
        }
        
        if (!this.playPauseButton) {
            console.error('Play/pause button not found');
        }
        
        // Bind event handlers
        this.handlePlayPause = this.handlePlayPause.bind(this);
    }

    /**
     * Initialize the controller by fetching configuration and setting up UI
     */
    async initialize() {
        // Show loading indicator
        this.showLoading('Loading configuration...');
        
        try {
            // Fetch configuration from backend with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
            
            const response = await fetch('/api/config', {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                let errorMessage = 'Failed to load configuration';
                let errorType = 'unknown';
                
                try {
                    const errorData = await response.json();
                    if (errorData.detail) {
                        if (typeof errorData.detail === 'object') {
                            errorMessage = errorData.detail.message || errorMessage;
                            errorType = errorData.detail.type || errorType;
                        } else {
                            errorMessage = errorData.detail;
                        }
                    }
                } catch (e) {
                    // If JSON parsing fails, use default message
                    console.error('Error parsing error response:', e);
                }
                
                const error = new Error(errorMessage);
                error.type = errorType;
                throw error;
            }
            
            const config = await response.json();
            this.features = config.features || [];
            
            if (this.features.length === 0) {
                throw new Error('No features found in configuration');
            }
            
            // Hide loading indicator
            this.hideLoading();
            
            // Render feature toggle buttons
            this.renderFeatureToggles(this.features);
            
            // Set up play/pause button event listener
            if (this.playPauseButton) {
                this.playPauseButton.addEventListener('click', this.handlePlayPause);
            }
            
            // Load first feature by default
            this.handleFeatureSwitch(this.features[0].id);
            
        } catch (error) {
            this.hideLoading();
            console.error('Error initializing UI:', error);
            
            // Determine error type and show appropriate message
            let errorMessage = 'Failed to load application configuration.';
            let showRetry = true;
            
            if (error.name === 'AbortError') {
                errorMessage = 'Request timed out. Please check your network connection and try again.';
            } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                errorMessage = 'Network error. Unable to connect to the server. Please check your connection and try again.';
            } else if (error.type === 'config_missing') {
                errorMessage = error.message || 'Configuration file is missing. Please contact support.';
                showRetry = false;
            } else if (error.type === 'config_invalid') {
                errorMessage = error.message || 'Configuration file is invalid. Please contact support.';
                showRetry = false;
            } else if (error.message.includes('No features found')) {
                errorMessage = 'No analysis features are configured. Please contact support.';
                showRetry = false;
            } else if (error.message) {
                errorMessage = error.message;
            }
            
            this.displayError(errorMessage, showRetry);
        }
    }

    /**
     * Dynamically create toggle buttons for each feature
     * @param {Array} features - Array of feature configuration objects
     */
    renderFeatureToggles(features) {
        if (!this.featureTogglesContainer) {
            return;
        }
        
        // Clear existing toggles
        this.featureTogglesContainer.innerHTML = '';
        
        // Create toggle button for each feature
        features.forEach(feature => {
            const button = document.createElement('button');
            button.className = 'feature-toggle';
            button.dataset.featureId = feature.id;
            button.textContent = feature.name;
            
            // Add click event listener
            button.addEventListener('click', () => {
                this.handleFeatureSwitch(feature.id);
            });
            
            this.featureTogglesContainer.appendChild(button);
        });
    }

    /**
     * Handle feature toggle switch
     * @param {string} featureId - ID of the feature to switch to
     */
    handleFeatureSwitch(featureId) {
        try {
            // Find the feature configuration
            const feature = this.features.find(f => f.id === featureId);
            
            if (!feature) {
                console.error(`Feature not found: ${featureId}`);
                this.displayError(`Feature "${featureId}" not found.`, false);
                return;
            }
            
            // Capture current playback timestamp
            const currentTime = this.videoPlayer.getCurrentTime();
            
            // Switch video source and resume at captured timestamp
            this.videoPlayer.setSource(feature.video_path, currentTime);
            
            // Update current feature state
            this.currentFeature = feature;
            
            // Update active toggle button styling
            this.updateActiveToggle(featureId);
            
            // Clear any existing errors
            this.clearError();
            
        } catch (error) {
            console.error('Error switching feature:', error);
            this.displayError('Failed to switch video. Please try again.', false);
        }
    }

    /**
     * Update active state of toggle buttons
     * @param {string} activeFeatureId - ID of the currently active feature
     */
    updateActiveToggle(activeFeatureId) {
        if (!this.featureTogglesContainer) {
            return;
        }
        
        // Remove active class from all toggles
        const allToggles = this.featureTogglesContainer.querySelectorAll('.feature-toggle');
        allToggles.forEach(toggle => {
            toggle.classList.remove('active');
        });
        
        // Add active class to selected toggle
        const activeToggle = this.featureTogglesContainer.querySelector(
            `[data-feature-id="${activeFeatureId}"]`
        );
        
        if (activeToggle) {
            activeToggle.classList.add('active');
        }
    }

    /**
     * Toggle video playback state
     */
    handlePlayPause() {
        if (!this.playPauseButton) {
            return;
        }
        
        const videoElement = this.videoPlayer.videoElement;
        
        if (videoElement.paused) {
            this.videoPlayer.play();
            this.updatePlayPauseButton(true);
        } else {
            this.videoPlayer.pause();
            this.updatePlayPauseButton(false);
        }
    }

    /**
     * Update play/pause button UI
     * @param {boolean} isPlaying - Whether video is currently playing
     */
    updatePlayPauseButton(isPlaying) {
        if (!this.playPauseButton) {
            return;
        }
        
        if (isPlaying) {
            this.playPauseButton.textContent = 'Pause';
            this.playPauseButton.classList.add('playing');
        } else {
            this.playPauseButton.textContent = 'Play';
            this.playPauseButton.classList.remove('playing');
        }
    }

    /**
     * Show loading indicator
     * @param {string} message - Loading message to display
     */
    showLoading(message = 'Loading...') {
        let loadingContainer = document.getElementById('loading-container');
        
        if (!loadingContainer) {
            loadingContainer = document.createElement('div');
            loadingContainer.id = 'loading-container';
            loadingContainer.className = 'loading-overlay';
            loadingContainer.innerHTML = `
                <div class="loading-content">
                    <div class="loading-spinner"></div>
                    <p class="loading-message">${message}</p>
                </div>
            `;
            document.body.appendChild(loadingContainer);
        } else {
            const messageElement = loadingContainer.querySelector('.loading-message');
            if (messageElement) {
                messageElement.textContent = message;
            }
            loadingContainer.style.display = 'flex';
        }
    }

    /**
     * Hide loading indicator
     */
    hideLoading() {
        const loadingContainer = document.getElementById('loading-container');
        if (loadingContainer) {
            loadingContainer.style.display = 'none';
        }
    }

    /**
     * Display error message to user
     * @param {string} message - Error message to display
     * @param {boolean} showRetry - Whether to show retry button
     */
    displayError(message, showRetry = false) {
        // Create error container if it doesn't exist
        let errorContainer = document.getElementById('error-container');
        
        if (!errorContainer) {
            errorContainer = document.createElement('div');
            errorContainer.id = 'error-container';
            errorContainer.className = 'error-message';
            document.body.insertBefore(errorContainer, document.body.firstChild);
        }
        
        // Build error content
        let errorHTML = `
            <div class="error-content">
                <span class="error-icon">⚠️</span>
                <span class="error-text">${message}</span>
        `;
        
        if (showRetry) {
            errorHTML += `
                <button class="retry-button" onclick="location.reload()">
                    Retry
                </button>
            `;
        }
        
        errorHTML += `
                <button class="close-button" aria-label="Close error message">×</button>
            </div>
        `;
        
        errorContainer.innerHTML = errorHTML;
        errorContainer.style.display = 'block';
        
        // Add close button handler
        const closeButton = errorContainer.querySelector('.close-button');
        if (closeButton) {
            closeButton.addEventListener('click', () => {
                this.clearError();
            });
        }
    }

    /**
     * Clear error message
     */
    clearError() {
        const errorContainer = document.getElementById('error-container');
        if (errorContainer) {
            errorContainer.style.display = 'none';
        }
    }
}
