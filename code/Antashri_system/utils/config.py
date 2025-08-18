# utils/config.py
"""
Configuration module for Antashiri System
Centralized configuration management
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


class Config:
    """
    Centralized configuration management for Antashiri
    Supports environment variables, JSON, and YAML configurations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or os.getenv('CONFIG_PATH', 'config/antashiri.yaml')
        self.config = self._load_config()
        self._apply_env_overrides()
        
        logger.info("Configuration loaded")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or defaults"""
        
        # Default configuration
        config = {
            'system': {
                'name': 'Antashiri',
                'version': '1.0.0',
                'debug': False
            },
            'models': {
                'emotion_detector': {
                    'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
                    'use_quantization': True,
                    'device': 'auto',
                    'cache_size': 100
                },
                'whisper': {
                    'model_size': 'base',
                    'language': 'en',
                    'device': 'auto'
                },
                'embeddings': {
                    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                    'batch_size': 32,
                    'use_gpu': False
                }
            },
            'audio': {
                'sample_rate': 16000,
                'channels': 1,
                'chunk_size': 1024,
                'silence_threshold': 500,
                'recording_duration': 5,
                'max_recording_duration': 30
            },
            'music': {
                'playback_duration': 45,
                'fade_in_duration': 2,
                'fade_out_duration': 3,
                'min_trigger_interval': 30,
                'search_limit': 10
            },
            'emotion': {
                'detection_threshold': 0.6,
                'stability_window': 10,
                'min_confidence': 0.5,
                'max_history': 10
            },
            'server': {
                'host': '0.0.0.0',
                'port': 5000,
                'cors_origins': '*',
                'secret_key': 'dev-secret-key-change-in-production'
            },
            'youtube': {
                'api_key': None,
                'max_results': 10,
                'video_category': '10',  # Music category
                'prefer_official': True
            },
            'paths': {
                'data': './data',
                'models': './models',
                'logs': './logs',
                'temp': './temp',
                'cache': './cache'
            },
            'logging': {
                'level': 'INFO',
                'file': 'antashiri.log',
                'max_size': '10MB',
                'backup_count': 5,
                'format': '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>'
            },
            'cache': {
                'enabled': True,
                'ttl': 3600,
                'max_size': 1000
            }
        }
        
        # Try to load from file
        if self.config_path and Path(self.config_path).exists():
            try:
                file_config = self._load_file_config(self.config_path)
                config = self._merge_configs(config, file_config)
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        return config
    
    def _load_file_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries"""
        
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        
        # Model configurations
        if os.getenv('MODEL_NAME'):
            self.config['models']['emotion_detector']['model_name'] = os.getenv('MODEL_NAME')
        if os.getenv('WHISPER_MODEL'):
            self.config['models']['whisper']['model_size'] = os.getenv('WHISPER_MODEL')
        if os.getenv('EMBEDDING_MODEL'):
            self.config['models']['embeddings']['model_name'] = os.getenv('EMBEDDING_MODEL')
        
        # Audio settings
        if os.getenv('AUDIO_SAMPLE_RATE'):
            self.config['audio']['sample_rate'] = int(os.getenv('AUDIO_SAMPLE_RATE'))
        if os.getenv('RECORDING_DURATION'):
            self.config['audio']['recording_duration'] = int(os.getenv('RECORDING_DURATION'))
        
        # Music settings
        if os.getenv('MUSIC_DURATION'):
            self.config['music']['playback_duration'] = int(os.getenv('MUSIC_DURATION'))
        if os.getenv('MIN_TRIGGER_INTERVAL'):
            self.config['music']['min_trigger_interval'] = int(os.getenv('MIN_TRIGGER_INTERVAL'))
        
        # Emotion settings
        if os.getenv('EMOTION_THRESHOLD'):
            self.config['emotion']['detection_threshold'] = float(os.getenv('EMOTION_THRESHOLD'))
        if os.getenv('MIN_CONFIDENCE'):
            self.config['emotion']['min_confidence'] = float(os.getenv('MIN_CONFIDENCE'))
        
        # Server settings
        if os.getenv('HOST'):
            self.config['server']['host'] = os.getenv('HOST')
        if os.getenv('PORT'):
            self.config['server']['port'] = int(os.getenv('PORT'))
        if os.getenv('SECRET_KEY'):
            self.config['server']['secret_key'] = os.getenv('SECRET_KEY')
        
        # YouTube API
        if os.getenv('YOUTUBE_API_KEY'):
            self.config['youtube']['api_key'] = os.getenv('YOUTUBE_API_KEY')
        
        # Debug mode
        if os.getenv('DEBUG_MODE'):
            self.config['system']['debug'] = os.getenv('DEBUG_MODE').lower() == 'true'
        
        # Logging
        if os.getenv('LOG_LEVEL'):
            self.config['logging']['level'] = os.getenv('LOG_LEVEL')
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key
        
        Args:
            key: Configuration key (e.g., 'models.whisper.model_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by dot-notation key
        
        Args:
            key: Configuration key
            value: Value to set
        """
        
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file"""
        
        save_path = path or self.config_path
        
        if not save_path:
            logger.warning("No save path specified")
            return
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.json':
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif save_path.suffix in ['.yaml', '.yml']:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {save_path}")
    
    def validate(self) -> bool:
        """Validate configuration"""
        
        required_keys = [
            'models.emotion_detector.model_name',
            'models.whisper.model_size',
            'audio.sample_rate',
            'server.port'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                logger.error(f"Required configuration missing: {key}")
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return self.config.copy()