#!/usr/bin/env python3
"""
Setup script for Deep-Live-Cam face swapping integration
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command with error handling"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def download_deep_live_cam(install_dir="./Deep-Live-Cam"):
    """Download Deep-Live-Cam repository"""
    if Path(install_dir).exists():
        print(f"Deep-Live-Cam already exists at {install_dir}")
        return True
    
    return run_command([
        'git', 'clone', 
        'https://github.com/hacksider/Deep-Live-Cam.git',
        install_dir
    ], "Downloading Deep-Live-Cam repository")

def install_face_swap_dependencies(cpu_only=False):
    """Install face swapping dependencies"""
    dependencies = [
        'insightface>=0.7.3',
        'onnx>=1.14.0',
    ]
    
    if cpu_only:
        dependencies.append('onnxruntime>=1.15.0')
        print("Installing CPU-only ONNX Runtime")
    else:
        # Try GPU version first, fallback to CPU
        gpu_install = run_command([
            'uv', 'pip', 'install', 
            'onnxruntime-gpu>=1.15.0'
        ], "Installing GPU ONNX Runtime")
        
        if not gpu_install:
            print("GPU ONNX Runtime failed, falling back to CPU version")
            dependencies.append('onnxruntime>=1.15.0')
    
    for dep in dependencies:
        if not run_command([
            'uv', 'pip', 'install', dep
        ], f"Installing {dep}"):
            return False
    
    return True

def setup_models_directory():
    """Create models directory structure"""
    models_dir = Path('./models/face_swap')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created models directory: {models_dir}")
    
    # Create a readme file
    readme_content = """# Face Swap Models Directory

This directory stores face swapping models for Deep-Live-Cam integration.

## Models that will be downloaded automatically:
- inswapper_128.onnx: Main face swapping model
- Additional InsightFace models as needed

## Manual model setup:
If automatic download fails, you can manually download models:
1. Download inswapper_128.onnx from the Deep-Live-Cam releases
2. Place it in this directory
3. Update the model path in config.json if needed
"""
    
    readme_path = models_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Created README: {readme_path}")
    return True

def update_config(enable_face_swap=False):
    """Update configuration to enable face swapping"""
    config_path = Path('./config.json')
    
    if not config_path.exists():
        print("⚠ config.json not found, skipping config update")
        return True
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update face swap settings
        if 'face_swap' not in config:
            config['face_swap'] = {}
        
        config['face_swap']['enabled'] = enable_face_swap
        config['face_swap']['device'] = 'auto'
        config['face_swap']['model_path'] = './models/face_swap'
        config['face_swap']['cache_reference_faces'] = True
        
        # Write back
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        status = "enabled" if enable_face_swap else "configured (disabled)"
        print(f"✓ Face swapping {status} in config.json")
        return True
        
    except Exception as e:
        print(f"✗ Failed to update config: {e}")
        return False

def test_installation():
    """Test the face swapping installation"""
    print("\nTesting face swapping installation...")
    
    test_script = """
import sys
try:
    import insightface
    print("✓ InsightFace imported successfully")
    
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"✓ ONNX Runtime providers: {providers}")
    
    # Test face analysis initialization
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    print("✓ Face analysis initialized")
    
    print("\\n🎉 Face swapping installation test PASSED!")
    sys.exit(0)
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Initialization error: {e}")
    sys.exit(1)
"""
    
    try:
        result = subprocess.run([
            'uv', 'run', 'python', '-c', test_script
        ], capture_output=True, text=True, timeout=30)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("✗ Test timed out (may be downloading models)")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Setup Deep-Live-Cam face swapping')
    parser.add_argument('--cpu-only', action='store_true', 
                       help='Install CPU-only versions (no GPU acceleration)')
    parser.add_argument('--enable', action='store_true',
                       help='Enable face swapping in config.json')
    parser.add_argument('--no-download', action='store_true',
                       help='Skip downloading Deep-Live-Cam repository')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run installation test')
    
    args = parser.parse_args()
    
    if args.test_only:
        success = test_installation()
        sys.exit(0 if success else 1)
    
    print("🚀 Setting up Deep-Live-Cam face swapping integration...")
    
    success = True
    
    # Step 1: Download repository
    if not args.no_download:
        success &= download_deep_live_cam()
    
    # Step 2: Install dependencies
    success &= install_face_swap_dependencies(cpu_only=args.cpu_only)
    
    # Step 3: Setup models directory
    success &= setup_models_directory()
    
    # Step 4: Update configuration
    success &= update_config(enable_face_swap=args.enable)
    
    # Step 5: Test installation
    if success:
        print("\n🧪 Testing installation...")
        test_success = test_installation()
        success &= test_success
    
    if success:
        print("\n🎉 Deep-Live-Cam face swapping setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the avatar mirror system: python src/main.py")
        if not args.enable:
            print("2. Enable face swapping in config.json or set FACE_SWAP_ENABLED=true")
        print("3. Upload reference faces via the web interface or place them in cache/face_swap/")
        print("4. Toggle face swapping for specific people as needed")
        
    else:
        print("\n❌ Setup failed. Check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()