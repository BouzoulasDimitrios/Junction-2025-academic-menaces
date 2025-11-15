# Drone Analysis Video Player

A web-based video player application for visualizing multiple analysis outputs from event camera recordings of drone movements. Built with FastAPI and vanilla JavaScript.

## Features

- **Synchronized Video Playback**: Switch between different analysis features while maintaining playback position
- **Continuous Looping**: Videos automatically restart for continuous observation
- **Dynamic Feature Toggles**: Easily add or remove analysis features through configuration
- **Visualization Support**: Display supplementary graphs and charts alongside videos
- **Junction Branding**: Customized with Junction hackathon branding
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone or download this repository

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Add your video files to the `videos/` directory (see `videos/README.md` for details)

4. Run the application:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

5. Open your browser and navigate to:
```
http://localhost:8000
```

## Project Structure

```
project/
├── main.py                          # FastAPI application
├── config.py                        # Configuration loader
├── features.json                    # Feature configuration (edit this to add/remove features)
├── requirements.txt                 # Python dependencies
├── videos/                          # Video files directory
│   └── README.md                    # Video file instructions
├── static/                          # Static assets
│   ├── css/
│   │   └── styles.css              # Application styles
│   ├── js/
│   │   ├── main.js                 # Application entry point
│   │   ├── video-player.js         # Video player component
│   │   ├── ui-controller.js        # UI controller component
│   │   └── visualization-manager.js # Visualization manager
│   ├── images/
│   │   └── junction-logo.svg       # Junction logo
│   └── visualizations/              # Visualization images
│       ├── rpm-chart.svg
│       └── speed-graph.svg
└── templates/
    └── index.html                   # Main HTML page
```

## Adding New Features

To add a new analysis feature to the application:

### 1. Add Your Video File

Place your video file in the `videos/` directory. Ensure it:
- Is in MP4 format (H.264 codec)
- Has the same duration as other videos
- Is optimized for web delivery

### 2. Update Configuration

Edit `features.json` and add a new feature object to the `features` array:

```json
{
  "id": "your-feature-id",
  "name": "Your Feature Name",
  "video_path": "videos/your-video.mp4",
  "visualizations": []
}
```

**Configuration Fields:**
- `id`: Unique identifier (use kebab-case, e.g., "propeller-speed")
- `name`: Display name shown in the UI
- `video_path`: Path to video file relative to project root
- `visualizations`: Array of visualization objects (optional)

### 3. Add Visualizations (Optional)

If your feature has associated graphs or charts:

1. Add visualization images to `static/visualizations/`
2. Update the feature configuration:

```json
{
  "id": "your-feature-id",
  "name": "Your Feature Name",
  "video_path": "videos/your-video.mp4",
  "visualizations": [
    {
      "id": "your-viz-id",
      "type": "chart",
      "image_path": "static/visualizations/your-chart.svg",
      "title": "Your Chart Title"
    }
  ]
}
```

### 4. Restart the Server

If the server is running with `--reload`, it should automatically pick up the changes. Otherwise, restart it manually.

### 5. Test Your Feature

- Open the application in your browser
- Verify the new feature toggle appears
- Click the toggle and verify the video plays
- Test that visualizations display correctly (if added)

## Example Configuration

Here's a complete example of a feature with visualization:

```json
{
  "id": "propeller-speed",
  "name": "Propeller Speed Analysis",
  "video_path": "videos/propeller-speed.mp4",
  "visualizations": [
    {
      "id": "rpm-chart",
      "type": "chart",
      "image_path": "static/visualizations/rpm-chart.svg",
      "title": "RPM Over Time"
    }
  ]
}
```

## Removing Features

To remove a feature:

1. Open `features.json`
2. Delete the feature object from the `features` array
3. Save the file
4. The application will automatically update (if running with `--reload`)

## Customization

### Branding

To customize the Junction branding:

1. Replace `static/images/junction-logo.svg` with your logo
2. Edit `static/css/styles.css` to update colors and styles
3. Look for CSS variables at the top of the file for easy color changes

### Styling

The application uses CSS variables for easy theming. Edit `static/css/styles.css`:

```css
:root {
  --primary-color: #FF6B6B;
  --secondary-color: #4ECDC4;
  /* ... other variables */
}
```

## Development

### Running in Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables auto-reload when code changes are detected.

### Production Deployment

For production, use Gunicorn with Uvicorn workers:

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Troubleshooting

### Videos Don't Play

- Check that video files exist in the `videos/` directory
- Verify filenames in `features.json` match actual files
- Ensure videos are in MP4 format with H.264 codec
- Check browser console for errors

### Feature Toggles Don't Appear

- Verify `features.json` is valid JSON (use a JSON validator)
- Check that the configuration file is in the project root
- Look for errors in the browser console or server logs

### Visualizations Don't Display

- Verify image files exist at the specified paths
- Check that image paths in `features.json` are correct
- Ensure image files are in a web-compatible format (SVG, PNG, JPG)

### Server Won't Start

- Verify Python dependencies are installed: `pip install -r requirements.txt`
- Check that port 8000 is not already in use
- Look for error messages in the terminal

## Browser Compatibility

Tested and supported on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

This project was created for Junction hackathon demonstrations.

## Support

For issues or questions, please refer to the project documentation or contact the development team.
