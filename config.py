"""
Configuration management module for loading and accessing model settings.
"""
import json
import os
from typing import Dict, Any

def load_config(config_path: str = "model_config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Dictionary containing all configurations
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

# Load configurations at module level
try:
    CONFIG = load_config()
    MODEL_CONFIGS = CONFIG['model_configs']
    TRAINING_CONFIG = CONFIG['training_config']
    DATASET_CONFIG = CONFIG['dataset_config']
except Exception as e:
    print(f"Error loading configuration: {e}")
    # Provide default configurations if loading fails
    CONFIG = {}
    MODEL_CONFIGS = {}
    TRAINING_CONFIG = {}
    DATASET_CONFIG = {}

__all__ = ['CONFIG', 'MODEL_CONFIGS', 'TRAINING_CONFIG', 'DATASET_CONFIG', 'load_config'] 