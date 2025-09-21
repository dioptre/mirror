"""Tests for main application"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.main import AvatarMirrorSystem


class TestAvatarMirrorSystem:
    """Test cases for AvatarMirrorSystem"""

    def test_init(self):
        """Test system initialization"""
        system = AvatarMirrorSystem()
        assert system is not None
        assert system.config is not None
        assert not system.running

    def test_setup_logging(self):
        """Test logging setup"""
        system = AvatarMirrorSystem()
        # Should not raise any exceptions
        system._setup_logging()

    @patch('cv2.VideoCapture')
    def test_initialize_camera_success(self, mock_video_capture):
        """Test successful camera initialization"""
        mock_camera = Mock()
        mock_camera.isOpened.return_value = True
        mock_camera.get.side_effect = [1920, 1080, 30.0]  # width, height, fps
        mock_video_capture.return_value = mock_camera

        system = AvatarMirrorSystem()
        result = system.initialize_camera()

        assert result is True
        assert system.camera is not None
        mock_camera.set.assert_called()

    @patch('cv2.VideoCapture')
    def test_initialize_camera_failure(self, mock_video_capture):
        """Test camera initialization failure"""
        mock_camera = Mock()
        mock_camera.isOpened.return_value = False
        mock_video_capture.return_value = mock_camera

        system = AvatarMirrorSystem()
        result = system.initialize_camera()

        assert result is False
        assert system.camera is not None  # Camera object exists but not opened

    def test_get_latest_face_box(self):
        """Test getting latest face box from person data"""
        system = AvatarMirrorSystem()

        # Mock person data with face history
        person_data = Mock()
        person_data.face_history = [(10, 20, 100, 150, 0.9)]

        result = system._get_latest_face_box(person_data)
        assert result == (10, 20, 100, 150)

        # Test with no face history
        person_data.face_history = []
        result = system._get_latest_face_box(person_data)
        assert result == (0, 0, 100, 100)  # Default fallback