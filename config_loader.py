"""
Configuration loader for TaxForecaster application.

This module handles loading and accessing configuration from config.yaml.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager for TaxForecaster application"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one config instance"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config.yaml file. If None, searches in current directory.
        """
        if self._config is None:
            if config_path is None:
                # Try to find config.yaml in current directory or parent
                current_dir = Path(__file__).parent
                config_path = current_dir / "config.yaml"
                
                if not config_path.exists():
                    # Try parent directory
                    config_path = current_dir.parent / "config.yaml"
            
            self.config_path = config_path
            self.load()
    
    def load(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            print(f"[INFO] Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            print(f"[WARNING] Config file not found: {self.config_path}. Using defaults.")
            self._config = self._get_default_config()
        except yaml.YAMLError as e:
            print(f"[ERROR] Error parsing config file: {e}. Using defaults.")
            self._config = self._get_default_config()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Path to config value (e.g., 'models.training.default_n_trials')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config = Config()
            >>> n_trials = config.get('models.training.default_n_trials', 10)
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict:
        """
        Get entire configuration section
        
        Args:
            section: Section name (e.g., 'models', 'data')
            
        Returns:
            Dictionary containing section configuration
        """
        return self._config.get(section, {})
    
    def reload(self):
        """Reload configuration from file"""
        self._config = None
        self.load()
    
    @property
    def all(self) -> Dict:
        """Get all configuration"""
        return self._config.copy()
    
    def _get_default_config(self) -> Dict:
        """Return minimal default configuration if file not found"""
        return {
            'app': {
                'name': 'TaxForecaster 2.0',
                'version': '2.0.0',
                'debug': False
            },
            'models': {
                'training': {
                    'default_n_trials': 10,
                    'default_epochs': 100
                },
                'features': {
                    'lag_periods': 3,
                    'rolling_window': 3
                }
            },
            'data': {
                'min_rows': 12,
                'date_format': '%d/%m/%Y'
            },
            'logging': {
                'level': 'INFO'
            }
        }


# Global config instance
config = Config()


# Convenience functions
def get(key_path: str, default: Any = None) -> Any:
    """Get configuration value"""
    return config.get(key_path, default)


def get_section(section: str) -> Dict:
    """Get configuration section"""
    return config.get_section(section)


def reload():
    """Reload configuration"""
    config.reload()


# Example usage
if __name__ == "__main__":
    # Test configuration loading
    cfg = Config()
    
    print("=" * 60)
    print("Configuration Test")
    print("=" * 60)
    
    print(f"\nApp Name: {cfg.get('app.name')}")
    print(f"Default Trials: {cfg.get('models.training.default_n_trials')}")
    print(f"Lag Periods: {cfg.get('models.features.lag_periods')}")
    
    print("\nModel Settings:")
    model_config = cfg.get_section('models')
    print(f"  Enabled Models: {len(model_config.get('enabled_models', []))}")
    print(f"  Default Epochs: {model_config['training']['default_epochs']}")
    
    print("\nData Settings:")
    data_config = cfg.get_section('data')
    print(f"  Min Rows: {data_config['min_rows']}")
    print(f"  Date Format: {data_config['date_format']}")
