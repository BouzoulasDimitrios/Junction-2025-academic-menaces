/**
 * VideoPlayer Component
 * Manages dual HTML5 video elements for smooth crossfade transitions
 */
class VideoPlayer {
    /**
     * Initialize VideoPlayer with dual video elements
     * @param {HTMLVideoElement} videoElement1 - First video element
     * @param {HTMLVideoElement} videoElement2 - Second video element
     */
    constructor(videoElement1, videoElement2) {
        if (!videoElement1 || !(videoElement1 instanceof HTMLVideoElement)) {
            throw new Error('Valid HTMLVideoElement required for video 1');
        }
        if (!videoElement2 || !(videoElement2 instanceof HTMLVideoElement)) {
            throw new Error('Valid HTMLVideoElement required for video 2');
        }
        
        this.video1 = videoElement1;
        this.video2 = videoElement2;
        this.activeVideo = this.video1;
        this.inactiveVideo = this.video2;
        this.isLoopEnabled = false;
        this.errorCallback = null;
        this.isTransitioning = false;
        
        // Set up ended event listeners for both videos
        [this.video1, this.video2].forEach(video => {
            video.addEventListener('ended', () => {
                if (this.isLoopEnabled && video === this.activeVideo) {
                    video.currentTime = 0;
                    video.play();
                }
            });
            
            video.addEventListener('error', (e) => {
                this.handleVideoError(e);
            });
        });
    }
    
    /**
     * Get the currently active video element
     * @returns {HTMLVideoElement}
     */
    get videoElement() {
        return this.activeVideo;
    }

    /**
     * Set error callback function
     * @param {Function} callback - Function to call when video error occurs
     */
    setErrorCallback(callback) {
        this.errorCallback = callback;
    }

    /**
     * Handle video loading errors
     * @param {Event} event - Error event
     */
    handleVideoError(event) {
        const error = this.videoElement.error;
        let errorMessage = 'Failed to load video.';
        
        if (error) {
            switch (error.code) {
                case error.MEDIA_ERR_ABORTED:
                    errorMessage = 'Video loading was aborted.';
                    break;
                case error.MEDIA_ERR_NETWORK:
                    errorMessage = 'Network error while loading video.';
                    break;
                case error.MEDIA_ERR_DECODE:
                    errorMessage = 'Video file is corrupted or in an unsupported format.';
                    break;
                case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                    errorMessage = 'Video format not supported by your browser.';
                    break;
                default:
                    errorMessage = 'Unknown error loading video.';
            }
        }
        
        console.error('Video error:', errorMessage, error);
        
        // Call error callback if set
        if (this.errorCallback) {
            this.errorCallback(errorMessage);
        }
    }

    /**
     * Start video playback
     */
    play() {
        this.activeVideo.play().catch(error => {
            console.error('Error playing video:', error);
            
            let errorMessage = 'Failed to play video.';
            
            if (error.name === 'NotAllowedError') {
                errorMessage = 'Playback not allowed. Please interact with the page first.';
            } else if (error.name === 'NotSupportedError') {
                errorMessage = 'Video format not supported.';
            }
            
            if (this.errorCallback) {
                this.errorCallback(errorMessage);
            }
        });
    }

    /**
     * Stop video playback
     */
    pause() {
        this.activeVideo.pause();
        this.inactiveVideo.pause();
    }

    /**
     * Get current playback timestamp
     * @returns {number} Current time in seconds
     */
    getCurrentTime() {
        return this.activeVideo.currentTime;
    }

    /**
     * Switch video source with smooth crossfade transition
     * @param {string} videoUrl - URL of the new video source
     * @param {number} startTime - Timestamp to resume playback (in seconds)
     */
    setSource(videoUrl, startTime = 0) {
        // Prevent multiple simultaneous transitions
        if (this.isTransitioning) {
            return;
        }
        
        // Check if already playing this video
        if (this.activeVideo.src.endsWith(videoUrl)) {
            return;
        }
        
        try {
            this.isTransitioning = true;
            const wasPlaying = !this.activeVideo.paused;
            
            // Set new video source on inactive video
            this.inactiveVideo.src = videoUrl;
            
            // Set up timeout for loading
            const loadTimeout = setTimeout(() => {
                console.error('Video loading timeout');
                this.isTransitioning = false;
                if (this.errorCallback) {
                    this.errorCallback('Video loading timed out. Please try again.');
                }
            }, 15000); // 15 second timeout
            
            // Wait for video to load enough data
            this.inactiveVideo.addEventListener('canplay', () => {
                clearTimeout(loadTimeout);
                
                try {
                    // Ensure startTime is within video duration
                    if (startTime > this.inactiveVideo.duration) {
                        startTime = 0;
                    }
                    
                    this.inactiveVideo.currentTime = startTime;
                    
                    // Start playing the new video if previous was playing
                    if (wasPlaying) {
                        this.inactiveVideo.play().then(() => {
                            // Perform crossfade
                            this.crossfade();
                        }).catch(error => {
                            console.error('Error playing new video:', error);
                            this.isTransitioning = false;
                        });
                    } else {
                        // Just crossfade without playing
                        this.crossfade();
                    }
                } catch (error) {
                    console.error('Error setting video time:', error);
                    this.isTransitioning = false;
                    if (this.errorCallback) {
                        this.errorCallback('Failed to set video position.');
                    }
                }
            }, { once: true });
            
            // Load the new video
            this.inactiveVideo.load();
            
        } catch (error) {
            console.error('Error setting video source:', error);
            this.isTransitioning = false;
            if (this.errorCallback) {
                this.errorCallback('Failed to load video source.');
            }
        }
    }
    
    /**
     * Perform smooth crossfade between active and inactive videos
     */
    crossfade() {
        // Swap CSS classes for fade effect
        this.activeVideo.classList.remove('active-video');
        this.activeVideo.classList.add('inactive-video');
        
        this.inactiveVideo.classList.remove('inactive-video');
        this.inactiveVideo.classList.add('active-video');
        
        // Wait for transition to complete
        setTimeout(() => {
            // Pause and clear the now-inactive video
            this.activeVideo.pause();
            
            // Swap references
            const temp = this.activeVideo;
            this.activeVideo = this.inactiveVideo;
            this.inactiveVideo = temp;
            
            this.isTransitioning = false;
        }, 500); // Match CSS transition duration
    }

    /**
     * Enable continuous loop playback
     */
    enableLoop() {
        this.isLoopEnabled = true;
    }
}
