"""FastAPI application for drone analysis video player."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List, Dict, Any
import logging

from config import load_config, FeatureConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Drone Analysis Video Player",
    description="Web application for viewing drone analysis videos with synchronized playback",
    version="1.0.0"
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
app.mount("/videos", StaticFiles(directory="videos"), name="videos")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load configuration at startup
config_cache: List[FeatureConfig] = []


@app.on_event("startup")
async def startup_event():
    """Load configuration when application starts."""
    global config_cache
    try:
        config_cache = load_config()
        logger.info(f"Loaded {len(config_cache)} features from configuration")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        # Don't fail startup, but log the error
        config_cache = []


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """
    Serve the main HTML page.
    
    Returns:
        HTML content of the main application page
    """
    try:
        html_path = Path("templates/index.html")
        if not html_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Main HTML page not found. Please ensure templates/index.html exists."
            )
        
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/config")
async def get_config() -> JSONResponse:
    """
    Get feature configuration as JSON.
    
    Returns:
        JSON response containing array of feature configurations
    """
    global config_cache
    
    try:
        if not config_cache:
            # Try to reload configuration if cache is empty
            try:
                config_cache = load_config()
            except FileNotFoundError as e:
                logger.error(f"Configuration file not found: {e}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Configuration file not found",
                        "message": "The features.json configuration file is missing. Please ensure it exists in the project root.",
                        "type": "config_missing"
                    }
                )
            except ValueError as e:
                logger.error(f"Invalid configuration: {e}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Invalid configuration",
                        "message": str(e),
                        "type": "config_invalid"
                    }
                )
        
        # Convert FeatureConfig objects to dictionaries
        features_data = []
        for feature in config_cache:
            feature_dict = {
                "id": feature.id,
                "name": feature.name,
                "video_path": feature.video_path,
                "visualizations": [
                    {
                        "id": viz.id,
                        "type": viz.type,
                        "image_path": viz.image_path,
                        "title": viz.title
                    }
                    for viz in feature.visualizations
                ]
            }
            features_data.append(feature_dict)
        
        return JSONResponse(content={"features": features_data})
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred while loading the configuration. Please check server logs.",
                "type": "server_error"
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
