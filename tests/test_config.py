"""Tests for configuration system"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from src.utils.config import Configuration


class TestConfiguration:
    """Test cases for Configuration class"""

    def test_default_config(self):
        """Test default configuration loading"""
        config = Configuration()
        
        # Test camera defaults
        assert config.get('camera', 'device_id') == 0
        assert config.get('camera', 'width') == 1920
        assert config.get('camera', 'height') == 1080
        assert config.get('camera', 'fps') == 30

        # Test face detection defaults
        assert config.get('face_detection', 'confidence_threshold') == 0.7
        
        # Test websocket defaults
        assert config.get('websocket', 'host') == 'localhost'
        assert config.get('websocket', 'port') == 8765

    def test_config_file_loading(self):
        """Test loading configuration from file"""
        # Create temporary config file
        test_config = {
            'camera': {'device_id': 1, 'width': 640, 'height': 480},
            'websocket': {'port': 9999}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_path = f.name

        try:
            config = Configuration(temp_path)
            
            # Test that custom values override defaults
            assert config.get('camera', 'device_id') == 1
            assert config.get('camera', 'width') == 640
            assert config.get('websocket', 'port') == 9999
            
            # Test that non-overridden values remain default
            assert config.get('camera', 'fps') == 30  # Default value
            
        finally:
            os.unlink(temp_path)

    def test_environment_variable_override(self):
        """Test environment variable configuration"""
        # Set environment variables
        os.environ['CAMERA_DEVICE_ID'] = '2'
        os.environ['WEBSOCKET_PORT'] = '8888'
        os.environ['USE_REDIS'] = 'true'
        
        try:
            config = Configuration()
            
            assert config.get('camera', 'device_id') == 2
            assert config.get('websocket', 'port') == 8888
            assert config.get('cache', 'use_redis') is True
            
        finally:
            # Clean up environment variables
            for var in ['CAMERA_DEVICE_ID', 'WEBSOCKET_PORT', 'USE_REDIS']:
                os.environ.pop(var, None)

    def test_config_validation(self):
        """Test configuration validation"""
        config = Configuration()
        
        # Test valid configuration
        issues = config.validate_config()
        assert len(issues) == 0  # Should have no issues with defaults
        
        # Test invalid camera device_id
        config.set('camera', 'device_id', -1)
        issues = config.validate_config()
        assert any('device_id must be >= 0' in issue for issue in issues)
        
        # Test invalid thresholds
        config.set('face_detection', 'confidence_threshold', 1.5)
        issues = config.validate_config()
        assert any('confidence threshold must be between 0.0 and 1.0' in issue for issue in issues)

    def test_save_config(self):
        """Test saving configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            config = Configuration()
            config.set('camera', 'device_id', 99)
            config.save_config(temp_path)
            
            # Load saved config
            with open(temp_path, 'r') as f:
                saved_config = json.load(f)
            
            assert saved_config['camera']['device_id'] == 99
            
        finally:
            os.unlink(temp_path)

    def test_get_summary(self):
        """Test configuration summary generation"""
        config = Configuration()
        summary = config.get_summary()
        
        assert 'Avatar Mirror Configuration Summary' in summary
        assert '[CAMERA]' in summary
        assert '[WEBSOCKET]' in summary
        assert 'device_id: 0' in summary