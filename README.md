# Avatar Mirror System

A complete real-time avatar mirror system that detects people entering a room, reconstructs them as 3D avatars, rigs the models for animation, and streams pose data to Godot for real-time visualization.

## Features

- **Face Detection & Tracking**: Ultra-light face detection with person tracking
- **Quality Assessment**: Ensures arms, face, and torso are visible before processing
- **Background Removal**: Clean background removal for better reconstruction
- **3D Reconstruction**: PIFuHD-based human reconstruction
- **Automatic Rigging**: UniRig integration for humanoid rigging
- **Pose Estimation**: TEMPO-based state-of-the-art pose estimation
- **Face Swapping**: Deep-Live-Cam integration for real-time face swapping
- **Real-time Streaming**: WebSocket communication with Godot
- **Intelligent Caching**: Redis and file-based caching for faces and models
- **Multi-threaded Processing**: Parallel processing for optimal performance
- **CPU/GPU Support**: Automatic device detection with CPU fallback

## System Requirements

### Hardware
- Camera (USB/built-in webcam)
- GPU recommended for 3D processing (CUDA support)
- Minimum 8GB RAM (16GB+ recommended)
- ~10GB free disk space for models and cache

### Software
- Python 3.8+
- OpenCV
- PyTorch (with CUDA support recommended)
- Redis (optional, for high-performance caching)

## Installation

### Prerequisites
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- Python 3.8+ 

Install uv:
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Quick Start Options

**🚀 For GPU Development (Recommended):**
```bash
git clone <repository-url>
cd avatar-mirror
make quick-start-gpu
```

**💻 For CPU-Only Development:**
```bash
git clone <repository-url>  
cd avatar-mirror
make quick-start-cpu
```

**🎭 For Full System (GPU + Face Swapping):**
```bash
git clone <repository-url>
cd avatar-mirror
make quick-start-full
```

### Manual Setup

1. **Clone and setup virtual environment**:
```bash
git clone <repository-url>
cd avatar-mirror
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies** (choose one):
```bash
# Basic installation
uv pip install -e .

# GPU acceleration
uv pip install -e ".[gpu,dev]"

# CPU-only (for development)  
uv pip install --index-url https://download.pytorch.org/whl/cpu -e ".[cpu,dev]"

# With face swapping
uv pip install -e ".[face-swap,dev]"

# Everything
uv pip install -e ".[all]"
```

3. **Setup AI Models**:
```bash
# Download all required AI models and repositories
make setup-models

# This automatically downloads:
# - OpenCV face detection models  
# - PIFuHD repository for 3D reconstruction
# - UniRig repository for rigging
# - TEMPO repository for pose estimation
# - Deep-Live-Cam repository for face swapping
```

4. **Optional: Setup Face Swapping**:
```bash
# Setup face swapping with uv
make setup-face-swap

# Or manually
uv run python scripts/setup_face_swap.py --enable
```

5. **Optional: Install Redis** (for better performance):
```bash
# On macOS
brew install redis

# On Ubuntu  
sudo apt-get install redis-server

# On Windows
# Download from https://redis.io/download
```

## Configuration

The system uses `config.json` for configuration. Key settings:

```json
{
  "camera": {
    "device_id": 0,
    "width": 1920,
    "height": 1080,
    "fps": 30
  },
  "websocket": {
    "host": "localhost",
    "port": 8765
  },
  "cache": {
    "use_redis": false,
    "cache_dir": "./cache"
  },
  "face_swap": {
    "enabled": false,
    "device": "auto"
  }
}
```

You can also use environment variables:
```bash
export CAMERA_DEVICE_ID=0
export WEBSOCKET_PORT=8765
export USE_REDIS=true
export FACE_SWAP_ENABLED=true
```

## Usage

### Basic Usage

1. **Start the system**:
```bash
# Using make (recommended)
make run

# Or directly with uv
uv run python src/main.py

# Or with activated environment  
source .venv/bin/activate
python src/main.py
```

2. **Connect from Godot**:
   - Create a WebSocket client in Godot
   - Connect to `ws://localhost:8765`
   - Listen for `new_character` and `pose_update` messages

### Advanced Usage

**Available make commands**:
```bash
make help              # Show all available commands
make setup-models      # Download all AI models
make run               # Run the system
make run-debug         # Run with debug logging
make test              # Run tests
make lint              # Check code quality
make format            # Format code
make clean             # Clean up generated files
```

**Debug mode**:
```bash
make run-debug
# Or set LOG_LEVEL=DEBUG in environment
```

**Face swapping**:
```bash
# Enable face swapping
export FACE_SWAP_ENABLED=true
make run

# Or setup face swapping completely
make setup-face-swap
```

## Godot Integration

### WebSocket Protocol

The system communicates with Godot via WebSocket using JSON messages:

#### Character Notification
```json
{
  "type": "new_character",
  "data": {
    "person_id": 1,
    "mesh_path": "/path/to/rigged_model.obj",
    "skeleton_data": {...},
    "joint_names": [...],
    "timestamp": "2023-..."
  }
}
```

#### Pose Updates
```json
{
  "type": "pose_update", 
  "data": {
    "1": {
      "pose_present": true,
      "joints": {
        "nose": {"position": [x, y, z], "confidence": 0.9},
        "left_shoulder": {"position": [x, y, z], "confidence": 0.8},
        ...
      },
      "confidence": 0.85,
      "timestamp": 1234567890
    }
  }
}
```

### Godot Client Example

```gdscript
extends Node

var websocket_client = WebSocketClient.new()

func _ready():
    websocket_client.connect_to_url("ws://localhost:8765")
    websocket_client.connect("connection_established", self, "_on_connected")
    websocket_client.connect("data_received", self, "_on_data_received")

func _on_connected():
    print("Connected to Avatar Mirror System")

func _on_data_received():
    var message = JSON.parse(websocket_client.get_peer(1).get_packet().get_string_from_utf8())
    
    match message.result.type:
        "new_character":
            _load_new_character(message.result.data)
        "pose_update":
            _update_poses(message.result.data)

func _load_new_character(character_data):
    # Load the 3D model and setup for animation
    var mesh_path = character_data.mesh_path
    var skeleton_data = character_data.skeleton_data
    # Implementation depends on your Godot setup

func _update_poses(pose_data):
    # Update character poses in real-time
    for person_id in pose_data:
        var pose = pose_data[person_id]
        if pose.pose_present:
            # Apply poses to your character
            pass
```

## System Architecture

```
Camera Input → Face Detection → Person Tracking → Quality Check
                                      ↓
Godot ← WebSocket ← Pose Estimation ← Background Removal
  ↑                                         ↓
Character         Cache ← Rigging ← 3D Reconstruction
Notification              Manager
```

## Performance Optimization

1. **GPU Acceleration**: Ensure CUDA is available for PyTorch
2. **Redis Caching**: Enable Redis for faster cache operations
3. **Worker Threads**: Adjust `num_workers` in config based on CPU cores
4. **Camera Resolution**: Lower resolution = faster processing
5. **Quality Threshold**: Higher threshold = fewer false positives

## Troubleshooting

### Common Issues

**Camera not detected**:
- Check `device_id` in config
- Ensure camera permissions are granted
- Try different device IDs (0, 1, 2...)

**Models not downloading**:
- Check internet connection
- Ensure sufficient disk space
- Models are downloaded to respective component directories

**Poor 3D reconstruction**:
- Ensure good lighting
- Person should be fully visible (arms, face, torso)
- Avoid cluttered backgrounds
- Check quality_threshold setting

**WebSocket connection issues**:
- Verify port is not in use
- Check firewall settings
- Ensure Godot client connects to correct address

### Performance Issues

**Low FPS**:
- Reduce camera resolution
- Lower face detection confidence threshold
- Reduce number of worker threads
- Enable GPU acceleration

**High memory usage**:
- Reduce cache sizes in config
- Enable cache cleanup intervals
- Lower model resolution settings

## Development

### Project Structure
```
avatar-mirror/
├── src/
│   ├── face_detection/      # Ultra-Light face detection
│   ├── tracking/            # Person tracking logic  
│   ├── background_removal/  # Background removal pipeline
│   ├── reconstruction/      # PIFuHD integration
│   ├── rigging/            # UniRig integration
│   ├── pose_estimation/    # TEMPO pose estimation
│   ├── face_swap/          # Deep-Live-Cam integration
│   ├── websocket/          # Godot communication
│   ├── cache/              # Caching system
│   ├── utils/              # Configuration and utilities
│   └── main.py             # Main application
├── tests/                  # Test suite
├── scripts/               # Setup and utility scripts
├── pyproject.toml        # Project configuration
├── Makefile             # Development commands
├── .venv/               # Virtual environment
└── README.md           # This file
```

### Development Workflow

1. **Setup development environment**:
```bash
make dev-setup  # Installs dev dependencies and pre-commit hooks
```

2. **Run tests**:
```bash
make test           # Run all tests
make test-cov      # Run tests with coverage
```

3. **Code quality**:
```bash
make format        # Format code with black/isort
make lint          # Check code quality  
make check         # Run all checks before commit
```

4. **Adding new features**:
   - Create new module in appropriate `src/` directory
   - Follow existing patterns for configuration
   - Add to `src/main.py` initialization sequence
   - Update `pyproject.toml` dependencies if needed
   - Add tests in `tests/` directory

### Testing

```bash
# Run all tests
make test

# Run with debug output
make run-debug

# Test specific functionality
uv run pytest tests/test_config.py -v
```

## License

See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## Support

For issues and questions:
1. Check troubleshooting section
2. Review logs in `avatar_mirror.log`
3. Open GitHub issue with full error details