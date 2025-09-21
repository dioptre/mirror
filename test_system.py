#!/usr/bin/env python3
"""
Simple test to verify Avatar Mirror System components without camera
"""

import sys
import os
sys.path.insert(0, 'src')

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing Avatar Mirror System imports...")
    
    try:
        from src.utils.config import config
        print("✅ Configuration system")
        
        from src.face_detection.detector import UltraLightFaceDetector
        print("✅ Face detection")
        
        from src.tracking.person_tracker import PersonTrackingSystem
        print("✅ Person tracking")
        
        from src.background_removal.processor import ImagePreprocessor
        print("✅ Background removal")
        
        from src.reconstruction.pifuhd_wrapper import ReconstructionPipeline
        print("✅ 3D reconstruction")
        
        from src.rigging.unirig_wrapper import RiggingPipeline
        print("✅ Model rigging")
        
        from src.pose_estimation.tempo_wrapper import PoseEstimationPipeline
        print("✅ Pose estimation")
        
        from src.websocket.godot_client import GodotIntegration
        print("✅ WebSocket communication")
        
        from src.cache.cache_manager import CacheManager
        print("✅ Caching system")
        
        print("\n🎉 All core components imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from src.utils.config import config
        
        # Test basic config access
        camera_device = config.get('camera', 'device_id')
        websocket_port = config.get('websocket', 'port')
        face_swap_enabled = config.get('face_swap', 'enabled')
        
        print(f"   Camera device: {camera_device}")
        print(f"   WebSocket port: {websocket_port}")
        print(f"   Face swap enabled: {face_swap_enabled}")
        
        # Test config validation
        issues = config.validate_config()
        if issues:
            print(f"   ⚠️ Configuration issues: {len(issues)}")
            for issue in issues[:3]:  # Show first 3 issues
                print(f"     - {issue}")
        else:
            print("   ✅ Configuration valid")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without camera"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test face detector creation
        from src.face_detection.detector import UltraLightFaceDetector
        detector = UltraLightFaceDetector()
        print("   ✅ Face detector created")
        
        # Test person tracker
        from src.tracking.person_tracker import PersonTrackingSystem
        tracker = PersonTrackingSystem()
        print("   ✅ Person tracker created")
        
        # Test cache manager
        from src.cache.cache_manager import CacheManager
        cache = CacheManager(cache_dir="./test_cache")
        print("   ✅ Cache manager created")
        
        # Test WebSocket integration
        from src.websocket.godot_client import GodotIntegration
        godot = GodotIntegration()
        print("   ✅ WebSocket integration created")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Avatar Mirror System - Component Test\n")
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_configuration():
        tests_passed += 1
        
    if test_basic_functionality():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your Avatar Mirror System is ready!")
        print("\n🚀 Next steps:")
        print("   1. Run the full system: make run")
        print("   2. Setup face swapping: make setup-face-swap")
        print("   3. Connect Godot client to ws://localhost:8765")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)