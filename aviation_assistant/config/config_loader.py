import os
import logging
from typing import Dict, Any, Optional
import yaml

class ConfigLoader:
    """
    Load and manage configuration for the application.
    
    Handles loading configuration from environment variables and files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = {}
        
        # Load configuration
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file and environment variables."""
        # Load from file if provided
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                self.logger.error(f"Error loading configuration from {self.config_path}: {str(e)}")
        
        # Override with environment variables
        env_config = {
            "avwx_api_key": os.getenv("AVWX_API_KEY"),
            "opensky_username": os.getenv("OPENSKY_USERNAME"),
            "opensky_password": os.getenv("OPENSKY_PASSWORD"),
            "faa_notam_api_key": os.getenv("FAA_NOTAM_API_KEY"),
            "flightaware_username": os.getenv("FLIGHTAWARE_USERNAME"),
            "flightaware_api_key": os.getenv("FLIGHTAWARE_API_KEY"),
            "windy_api_key": os.getenv("WINDY_API_KEY"),
            "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
            "data_cache_path": os.getenv("DATA_CACHE_PATH"),
            "vector_db_path": os.getenv("VECTOR_DB_PATH"),
            "log_level": os.getenv("LOG_LEVEL", "INFO")
        }
        
        # Remove None values
        env_config = {k: v for k, v in env_config.items() if v is not None}
        
        # Update config with environment variables
        self.config.update(env_config)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
