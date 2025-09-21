from setuptools import setup, find_packages

setup(
    name="avatar-mirror",
    version="1.0.0",
    description="Real-time avatar mirror system using 3D reconstruction and pose estimation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pillow>=8.3.0",
        "trimesh",
        "scipy",
        "scikit-image",
        "matplotlib",
        "open3d",
        "mediapipe",
        "websockets",
        "redis",
        "backgroundremover",
        "rembg",
    ],
)