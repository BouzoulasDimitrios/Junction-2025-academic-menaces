"""Configuration loader for drone analysis video player."""

import json
import os
from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class Visualization:
    """Represents a visualization (chart or graph) associated with a feature."""
    id: str
    type: str  # "chart" or "graph"
    image_path: str
    title: str

    @classmethod
    def from_dict(cls, data: dict) -> 'Visualization':
        """Create a Visualization instance from a dictionary."""
        return cls(
            id=data['id'],
            type=data['type'],
            image_path=data['image_path'],
            title=data['title']
        )


@dataclass
class FeatureConfig:
    """Represents a single analysis feature configuration."""
    id: str
    name: str
    video_path: str
    visualizations: List[Visualization]

    @classmethod
    def from_dict(cls, data: dict) -> 'FeatureConfig':
        """Create a FeatureConfig instance from a dictionary."""
        visualizations = [
            Visualization.from_dict(viz) for viz in data.get('visualizations', [])
        ]
        return cls(
            id=data['id'],
            name=data['name'],
            video_path=data['video_path'],
            visualizations=visualizations
        )


def load_config(config_path: str = 'features.json') -> List[FeatureConfig]:
    """
    Load and validate feature configuration from JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        List of FeatureConfig objects
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid or video files are missing
        json.JSONDecodeError: If JSON is malformed
    """
    # Check if configuration file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found. "
            f"Please ensure the file exists in the project root directory."
        )
    
    # Load and parse JSON
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON format in configuration file at line {e.lineno}, column {e.colno}: {e.msg}"
        )
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Configuration file encoding error: {e}. Please ensure the file is UTF-8 encoded."
        )
    except Exception as e:
        raise ValueError(f"Error reading configuration file: {e}")
    
    # Validate configuration structure
    if 'features' not in config_data:
        raise ValueError(
            "Configuration file is missing the 'features' array. "
            "Please add a 'features' key with an array of feature objects."
        )
    
    if not isinstance(config_data['features'], list):
        raise ValueError(
            "The 'features' key must contain an array of feature objects, "
            f"but found {type(config_data['features']).__name__} instead."
        )
    
    if len(config_data['features']) == 0:
        raise ValueError(
            "Configuration must contain at least one feature. "
            "Please add feature definitions to the 'features' array."
        )
    
    # Parse features and validate video files
    features = []
    missing_videos = []
    validation_errors = []
    
    for idx, feature_data in enumerate(config_data['features']):
        # Validate required fields
        required_fields = ['id', 'name', 'video_path']
        missing_fields = [field for field in required_fields if field not in feature_data]
        
        if missing_fields:
            validation_errors.append(
                f"Feature at index {idx} is missing required fields: {', '.join(missing_fields)}"
            )
            continue
        
        try:
            # Create feature config
            feature = FeatureConfig.from_dict(feature_data)
            
            # Validate video file exists
            if not os.path.exists(feature.video_path):
                missing_videos.append(f"{feature.name} ({feature.video_path})")
            
            features.append(feature)
            
        except Exception as e:
            validation_errors.append(f"Error parsing feature '{feature_data.get('name', 'unknown')}': {e}")
    
    # Report validation errors
    if validation_errors:
        raise ValueError(
            "Configuration validation errors:\n" + "\n".join(f"  - {err}" for err in validation_errors)
        )
    
    # Report missing video files
    if missing_videos:
        raise ValueError(
            f"Video files not found for the following features:\n" +
            "\n".join(f"  - {video}" for video in missing_videos) +
            "\n\nPlease ensure all video files exist at the specified paths."
        )
    
    return features
