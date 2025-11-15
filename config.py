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


@dataclass
class DatasetConfig:
    """Represents a dataset grouping that exposes a set of features."""
    id: str
    name: str
    features: List[FeatureConfig]

    @classmethod
    def from_dict(cls, data: dict) -> 'DatasetConfig':
        """Create a DatasetConfig instance from a dictionary."""
        features = [
            FeatureConfig.from_dict(feature_data)
            for feature_data in data.get('features', [])
        ]
        return cls(
            id=data['id'],
            name=data['name'],
            features=features
        )


def load_config(config_path: str = 'features.json') -> List[DatasetConfig]:
    """
    Load and validate feature configuration from JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        List of DatasetConfig objects
        
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
    
    # Determine datasets/feature structure
    if 'datasets' in config_data:
        datasets_raw = config_data['datasets']
        if not isinstance(datasets_raw, list) or len(datasets_raw) == 0:
            raise ValueError(
                "The 'datasets' key must contain a non-empty array of dataset objects."
            )
    elif 'features' in config_data:
        if not isinstance(config_data['features'], list) or len(config_data['features']) == 0:
            raise ValueError(
                "The 'features' key must contain a non-empty array of feature objects."
            )
        datasets_raw = [{
            "id": "default-dataset",
            "name": "Available Features",
            "features": config_data['features']
        }]
    else:
        raise ValueError(
            "Configuration file must include either a 'datasets' array or a 'features' array."
        )
    
    # Parse datasets/features and validate video files
    datasets = []
    missing_videos = []
    validation_errors = []
    
    for dataset_idx, dataset_data in enumerate(datasets_raw):
        # Validate dataset structure
        dataset_required_fields = ['id', 'name', 'features']
        missing_dataset_fields = [
            field for field in dataset_required_fields if field not in dataset_data
        ]
        
        if missing_dataset_fields:
            validation_errors.append(
                f"Dataset at index {dataset_idx} is missing required fields: "
                f"{', '.join(missing_dataset_fields)}"
            )
            continue
        
        if not isinstance(dataset_data.get('features'), list):
            validation_errors.append(
                f"Dataset '{dataset_data.get('name', 'unknown')}' must include a 'features' array."
            )
            continue
        
        dataset_features = []
        if len(dataset_data['features']) == 0:
            validation_errors.append(
                f"Dataset '{dataset_data.get('name', 'unknown')}' must contain at least one feature."
            )
            continue
        
        for idx, feature_data in enumerate(dataset_data['features']):
            # Validate required fields
            required_fields = ['id', 'name', 'video_path']
            missing_fields = [field for field in required_fields if field not in feature_data]
            
            if missing_fields:
                validation_errors.append(
                    f"Feature at index {idx} in dataset '{dataset_data['name']}' is missing "
                    f"required fields: {', '.join(missing_fields)}"
                )
                continue
            
            try:
                # Create feature config
                feature = FeatureConfig.from_dict(feature_data)
                
                # Validate video file exists
                if not os.path.exists(feature.video_path):
                    missing_videos.append(
                        f"{feature.name} ({feature.video_path}) in dataset '{dataset_data['name']}'"
                    )
                
                dataset_features.append(feature)
                
            except Exception as e:
                validation_errors.append(
                    f"Error parsing feature '{feature_data.get('name', 'unknown')}' "
                    f"in dataset '{dataset_data['name']}': {e}"
                )
        
        if dataset_features:
            datasets.append(
                DatasetConfig(
                    id=dataset_data['id'],
                    name=dataset_data['name'],
                    features=dataset_features
                )
            )
    
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
    
    return datasets
