/**
 * VisualizationManager - Manages display of supplementary visualizations (graphs/charts)
 * Requirements: 7.1, 7.2, 7.3, 7.4
 */
class VisualizationManager {
    /**
     * Initialize the VisualizationManager
     * @param {HTMLElement} containerElement - The container element for visualizations
     */
    constructor(containerElement) {
        this.container = containerElement;
        this.activeVisualizations = new Map(); // Map of visualizationId -> element
    }

    /**
     * Show a visualization with image and title
     * @param {Object} visualization - Visualization object with id, image_path, title, type
     */
    showVisualization(visualization) {
        // Check if visualization already exists
        if (this.activeVisualizations.has(visualization.id)) {
            const existingElement = this.activeVisualizations.get(visualization.id);
            existingElement.classList.remove('hide');
            existingElement.classList.add('show');
            return;
        }

        // Create visualization container
        const vizContainer = document.createElement('div');
        vizContainer.className = 'visualization-item';
        vizContainer.id = `viz-${visualization.id}`;
        vizContainer.setAttribute('data-viz-id', visualization.id);

        // Create header with title and toggle button
        const header = document.createElement('div');
        header.className = 'visualization-header';

        const title = document.createElement('h3');
        title.className = 'visualization-title';
        title.textContent = visualization.title;

        const toggleButton = document.createElement('button');
        toggleButton.className = 'visualization-toggle-btn';
        toggleButton.textContent = 'Hide';
        toggleButton.setAttribute('aria-label', `Hide ${visualization.title}`);
        toggleButton.addEventListener('click', () => {
            this.hideVisualization(visualization.id);
        });

        header.appendChild(title);
        header.appendChild(toggleButton);

        // Create image container
        const imageContainer = document.createElement('div');
        imageContainer.className = 'visualization-content';

        const image = document.createElement('img');
        image.src = visualization.image_path;
        image.alt = visualization.title;
        image.className = 'visualization-image';
        image.onerror = () => {
            imageContainer.innerHTML = `<p class="visualization-error">Failed to load visualization: ${visualization.title}</p>`;
        };

        imageContainer.appendChild(image);

        // Assemble visualization
        vizContainer.appendChild(header);
        vizContainer.appendChild(imageContainer);

        // Add to container and track
        this.container.appendChild(vizContainer);
        this.activeVisualizations.set(visualization.id, vizContainer);
        
        // Trigger animation by adding 'show' class after a brief delay
        setTimeout(() => {
            vizContainer.classList.add('show');
        }, 10);
    }

    /**
     * Hide a specific visualization
     * @param {string} visualizationId - The ID of the visualization to hide
     */
    hideVisualization(visualizationId) {
        const vizElement = this.activeVisualizations.get(visualizationId);
        if (vizElement) {
            vizElement.classList.remove('show');
            vizElement.classList.add('hide');
        }
    }

    /**
     * Remove all visualizations from display
     */
    clearAll() {
        // Remove all visualization elements from DOM
        this.activeVisualizations.forEach((element) => {
            element.remove();
        });
        
        // Clear the tracking map
        this.activeVisualizations.clear();
        
        // Clear container content
        this.container.innerHTML = '';
    }
}
