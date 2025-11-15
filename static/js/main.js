/**
 * Main Application Entry Point
 * Integrates VideoPlayer, VisualizationManager, and UIController components
 * Requirements: 4.5, 7.4, 7.5
 */

// Wait for DOM to be fully loaded before initializing
document.addEventListener('DOMContentLoaded', () => {
    try {
        // Get DOM element references
        const videoElement1 = document.getElementById('video-player-1');
        const videoElement2 = document.getElementById('video-player-2');
        const visualizationsContainer = document.getElementById('visualizations-container');
        
        // Validate required elements exist
        if (!videoElement1 || !videoElement2) {
            throw new Error('Video player elements not found');
        }
        
        if (!visualizationsContainer) {
            throw new Error('Visualizations container element not found');
        }
        
        // Instantiate VideoPlayer with dual video elements for smooth transitions
        const videoPlayer = new VideoPlayer(videoElement1, videoElement2);
        
        // Enable loop functionality for continuous playback
        videoPlayer.enableLoop();
        
        // Instantiate VisualizationManager with container element
        const visualizationManager = new VisualizationManager(visualizationsContainer);
        
        // Instantiate UIController with VideoPlayer instance
        const uiController = new UIController(videoPlayer);
        
        // Set up video error callback to display errors through UI controller
        videoPlayer.setErrorCallback((errorMessage) => {
            uiController.displayError(errorMessage, false);
        });
        
        // Store visualizationManager reference in uiController for feature switching
        uiController.visualizationManager = visualizationManager;
        
        // Override handleFeatureSwitch to include visualization updates
        const originalHandleFeatureSwitch = uiController.handleFeatureSwitch.bind(uiController);
        uiController.handleFeatureSwitch = function(featureId) {
            // Call original feature switch logic
            originalHandleFeatureSwitch(featureId);
            
            // Update visualizations for the new feature
            updateVisualizationsForFeature(this.currentFeature);
        };
        
        /**
         * Update visualizations when feature changes
         * @param {Object} feature - The currently active feature
         */
        function updateVisualizationsForFeature(feature) {
            if (!feature) {
                return;
            }
            
            // Clear all existing visualizations
            visualizationManager.clearAll();
            
            // Show visualizations for the current feature
            if (feature.visualizations && feature.visualizations.length > 0) {
                feature.visualizations.forEach(visualization => {
                    visualizationManager.showVisualization(visualization);
                });
            }
        }
        
        // Initialize the application
        uiController.initialize().then(() => {
            console.log('Drone Analysis Video Player initialized successfully');
        }).catch(error => {
            console.error('Failed to initialize application:', error);
        });
        
    } catch (error) {
        console.error('Error during application initialization:', error);
        
        // Display error to user
        const errorContainer = document.createElement('div');
        errorContainer.className = 'error-message';
        errorContainer.textContent = 'Failed to initialize application. Please refresh the page.';
        document.body.insertBefore(errorContainer, document.body.firstChild);
    }
});
