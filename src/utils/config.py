import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class Configuration:
    """Configuration management for the avatar mirror system"""
    
    DEFAULT_CONFIG = {
        # Camera settings
        'camera': {
            'device_id': 0,
            'width': 1920,
            'height': 1080,
            'fps': 30
        },
        
        # Face detection settings
        'face_detection': {
            'confidence_threshold': 0.7,
            'nms_threshold': 0.3,
            'model_path': None
        },
        
        # Person tracking settings
        'tracking': {
            'max_disappeared_frames': 10,
            'confirmation_frames': 5,
            'quality_threshold': 0.5,
            'face_cache_size': 20
        },
        
        # Background removal settings
        'background_removal': {
            'method': 'auto',  # 'rembg', 'backgroundremover', 'opencv', 'auto'
            'model_name': 'u2net_human_seg'
        },
        
        # PIFuHD settings
        'reconstruction': {
            'device': 'auto',
            'resolution': 512,
            'generate_color': True,
            'pifuhd_path': None,
            'checkpoint_path': None
        },
        
        # UniRig settings
        'rigging': {
            'device': 'auto',
            'rig_type': 'humanoid',
            'unirig_path': None
        },
        
        # TEMPO settings
        'pose_estimation': {
            'device': 'auto',
            'tempo_path': None,
            'fps': 30,
            'smoothing_window': 5
        },
        
        # WebSocket settings
        'websocket': {
            'host': 'localhost',
            'port': 8765,
            'max_clients': 10
        },
        
        # Cache settings
        'cache': {
            'use_redis': True,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0,
            'cache_dir': './cache',
            'max_face_cache': 20,
            'max_model_cache': 50
        },
        
        # Processing settings
        'processing': {
            'num_workers': 4,
            'process_timeout': 30,
            'enable_gpu_acceleration': True
        },
        
        # Face swapping settings (Deep-Live-Cam)
        'face_swap': {
            'enabled': False,
            'device': 'auto',
            'deep_live_cam_path': None,
            'model_path': './models/face_swap',
            'cache_reference_faces': True
        },
        
        # Logging settings
        'logging': {
            'level': 'INFO',
            'file': './avatar_mirror.log',
            'max_file_size': '10MB',
            'backup_count': 5
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or './config.json'
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load configuration from file if it exists
        self._load_config()
        
        # Override with environment variables
        self._load_from_environment()
    
    def _load_config(self):
        """Load configuration from file"""
        config_path = Path(self.config_file)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Merge with default config
                self._merge_config(self.config, file_config)
                logger.info(f"Loaded configuration from {config_path}")
                
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info("No configuration file found, using defaults")
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'CAMERA_DEVICE_ID': ('camera', 'device_id', int),
            'CAMERA_WIDTH': ('camera', 'width', int),
            'CAMERA_HEIGHT': ('camera', 'height', int),
            'CAMERA_FPS': ('camera', 'fps', int),
            
            'FACE_CONFIDENCE_THRESHOLD': ('face_detection', 'confidence_threshold', float),
            'FACE_MODEL_PATH': ('face_detection', 'model_path', str),
            
            'TRACKING_MAX_DISAPPEARED': ('tracking', 'max_disappeared_frames', int),
            'TRACKING_QUALITY_THRESHOLD': ('tracking', 'quality_threshold', float),
            
            'BG_REMOVAL_METHOD': ('background_removal', 'method', str),
            'BG_REMOVAL_MODEL': ('background_removal', 'model_name', str),
            
            'PIFUHD_DEVICE': ('reconstruction', 'device', str),
            'PIFUHD_PATH': ('reconstruction', 'pifuhd_path', str),
            'PIFUHD_CHECKPOINT': ('reconstruction', 'checkpoint_path', str),
            
            'UNIRIG_PATH': ('rigging', 'unirig_path', str),
            
            'TEMPO_PATH': ('pose_estimation', 'tempo_path', str),
            'POSE_FPS': ('pose_estimation', 'fps', int),
            
            'FACE_SWAP_ENABLED': ('face_swap', 'enabled', lambda x: x.lower() == 'true'),
            'FACE_SWAP_DEVICE': ('face_swap', 'device', str),
            'DEEP_LIVE_CAM_PATH': ('face_swap', 'deep_live_cam_path', str),
            
            'WEBSOCKET_HOST': ('websocket', 'host', str),
            'WEBSOCKET_PORT': ('websocket', 'port', int),
            
            'USE_REDIS': ('cache', 'use_redis', lambda x: x.lower() == 'true'),
            'REDIS_HOST': ('cache', 'redis_host', str),
            'REDIS_PORT': ('cache', 'redis_port', int),
            'CACHE_DIR': ('cache', 'cache_dir', str),
            
            'LOG_LEVEL': ('logging', 'level', str),
            'LOG_FILE': ('logging', 'file', str),
        }
        
        for env_var, (section, key, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    self.config[section][key] = converted_value
                    logger.debug(f"Set {section}.{key} = {converted_value} from {env_var}")
                except Exception as e:
                    logger.warning(f"Failed to convert {env_var}={value}: {e}")
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Merge configuration dictionaries recursively"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            section: Configuration section
            key: Key within section (optional)
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        if section not in self.config:
            return default
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key, default)
    
    def set(self, section: str, key: str, value: Any):
        """Set configuration value"""
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def save_config(self, config_file: Optional[str] = None):
        """Save current configuration to file"""
        save_path = config_file or self.config_file
        
        try:
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration and return list of issues
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Validate camera settings
        camera_config = self.get('camera')
        if camera_config['device_id'] < 0:
            issues.append("Camera device_id must be >= 0")
        
        if camera_config['width'] <= 0 or camera_config['height'] <= 0:
            issues.append("Camera width and height must be > 0")
        
        if camera_config['fps'] <= 0:
            issues.append("Camera fps must be > 0")
        
        # Validate thresholds
        face_config = self.get('face_detection')
        if not (0.0 <= face_config['confidence_threshold'] <= 1.0):
            issues.append("Face confidence threshold must be between 0.0 and 1.0")
        
        tracking_config = self.get('tracking')
        if not (0.0 <= tracking_config['quality_threshold'] <= 1.0):
            issues.append("Tracking quality threshold must be between 0.0 and 1.0")
        
        # Validate paths
        for section, key in [
            ('reconstruction', 'pifuhd_path'),
            ('rigging', 'unirig_path'),
            ('pose_estimation', 'tempo_path')
        ]:
            path = self.get(section, key)
            if path and not os.path.exists(path):
                issues.append(f"{section}.{key}: Path does not exist: {path}")
        
        # Validate cache settings
        cache_config = self.get('cache')
        if cache_config['max_face_cache'] <= 0:
            issues.append("max_face_cache must be > 0")
        
        if cache_config['max_model_cache'] <= 0:
            issues.append("max_model_cache must be > 0")
        
        # Validate websocket settings
        websocket_config = self.get('websocket')
        if not (1024 <= websocket_config['port'] <= 65535):
            issues.append("WebSocket port must be between 1024 and 65535")
        
        return issues
    
    def get_summary(self) -> str:
        """Get configuration summary"""
        summary = []
        summary.append("Avatar Mirror Configuration Summary:")
        summary.append("=" * 40)
        
        for section, config in self.config.items():
            summary.append(f"\n[{section.upper()}]")
            if isinstance(config, dict):
                for key, value in config.items():
                    summary.append(f"  {key}: {value}")
            else:
                summary.append(f"  {config}")
        
        return "\n".join(summary)


# Global configuration instance
config = Configuration()