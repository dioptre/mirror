# Godot Setup Guide for Avatar Mirror System

Complete step-by-step guide to get your Avatar Mirror working in Godot.

## Prerequisites

✅ Avatar Mirror System running (from main README)
✅ Godot 4.3+ installed

## Step 1: Setup Avatar Mirror System

```bash
# If you haven't already done this:
cd /Users/andrewgrosser/Documents/mirror

# Make sure all models are downloaded
make setup-models

# Start the Avatar Mirror System
make run
```

You should see output like:
```
✅ Connected to Avatar Mirror System
🤖 Checking and downloading required AI models...
📦 Setting up AI model repositories...
🎥 Camera initialized: 1920x1080 @ 30.0fps
```

## Step 2: Setup Godot Project

### Option A: Import Existing Project

1. **Open Godot 4.3+**
2. **Import Project**:
   - Click "Import" 
   - Navigate to `/Users/andrewgrosser/Documents/mirror/godot_client/`
   - Select `project.godot`
   - Click "Import & Edit"

### Option B: Create New Project

1. **Create new Godot project**
2. **Copy the client files** to your project:
   ```bash
   cp /Users/andrewgrosser/Documents/mirror/godot_client/*.gd your_project/
   ```
3. **Add AvatarMirrorClient** to your main scene

## Step 3: Test Connection

1. **Run the demo scene**:
   - Press **F5** in Godot
   - Select `MirrorDemo.tscn` as main scene
   - Click "Select Current"

2. **Check connection**:
   - You should see "✅ Connected to Avatar Mirror" in the debug panel
   - If connection fails, check that Avatar Mirror is running on port 8765

## Step 4: Test Avatar Creation

1. **Stand in front of the camera** connected to your Avatar Mirror System
2. **Wait for detection**:
   - System detects your face
   - Tracks you until it gets a good full-body view
   - Creates your 3D avatar (this takes ~30 seconds)
   - Sends avatar to Godot

3. **See your avatar**:
   - New avatar appears in Godot scene
   - Real-time pose updates from your movements
   - Debug panel shows "Avatars: 1"

## Step 5: Integration with Your Project

### Basic Integration

```gdscript
# In your main scene script
extends Node3D

var mirror_client: AvatarMirrorClient

func _ready():
    # Add avatar mirror client
    mirror_client = AvatarMirrorClient.new()
    add_child(mirror_client)
    
    # Connect signals
    mirror_client.character_received.connect(_on_new_character)
    mirror_client.pose_updated.connect(_on_pose_update)

func _on_new_character(character_data: Dictionary):
    var person_id = character_data.person_id
    print("New character: Person ", person_id)
    
    # The client automatically creates avatars
    # You can get the avatar node:
    var avatar = mirror_client.get_avatar(person_id)
    
    # Apply your custom setup
    if avatar:
        setup_custom_avatar(avatar, character_data)

func _on_pose_update(person_id: int, pose_data: Dictionary):
    # Pose updates are automatic
    # You can add custom effects here
    var avatar = mirror_client.get_avatar(person_id)
    if avatar and pose_data.pose_present:
        var confidence = pose_data.confidence
        # Add effects based on pose confidence, etc.
```

### Advanced Integration

```gdscript
# Custom avatar with your own mesh and animations
extends Avatar

@export var custom_mesh_scene: PackedScene
@export var animation_player: AnimationPlayer

func setup_character(char_data: Dictionary):
    super.setup_character(char_data)
    
    # Load your custom mesh
    if custom_mesh_scene:
        var custom_mesh = custom_mesh_scene.instantiate()
        add_child(custom_mesh)
    
    # Setup custom animations
    if animation_player:
        animation_player.play("idle")

func update_pose(pose_data: Dictionary):
    super.update_pose(pose_data)
    
    # Custom pose processing
    var joints = pose_data.get("joints", {})
    
    # Map to your animation system
    if joints.has("left_wrist") and joints.has("right_wrist"):
        var left_wrist = joints["left_wrist"]["position"]
        var right_wrist = joints["right_wrist"]["position"]
        
        # Check if hands are raised
        if left_wrist[1] > 0.5 and right_wrist[1] > 0.5:
            if animation_player:
                animation_player.play("hands_up")
```

## Message Protocol Reference

### Character Data Structure
```gdscript
{
    "person_id": 1,
    "mesh_path": "/path/to/model.obj",
    "skeleton_data": {
        "joint_positions": {...},
        "bone_hierarchy": {...}
    },
    "joint_names": ["root", "pelvis", "spine_01", ...],
    "timestamp": "2023-..."
}
```

### Pose Data Structure
```gdscript
{
    "1": {  # person_id
        "pose_present": true,
        "joints": {
            "nose": {"position": [x, y, z], "confidence": 0.9},
            "left_shoulder": {"position": [x, y, z], "confidence": 0.8},
            "right_shoulder": {"position": [x, y, z], "confidence": 0.8},
            # ... all joints
        },
        "confidence": 0.85,
        "timestamp": 1234567890
    }
}
```

## Troubleshooting

### "Connection Failed"
```bash
# Check Avatar Mirror is running
make run

# Check port is correct (default 8765)
netstat -an | grep 8765
```

### "No Avatars Appearing"
1. **Check camera**: Make sure someone is in front of the Avatar Mirror camera
2. **Check detection**: Look for "Person X detected" in Avatar Mirror logs
3. **Check processing**: Wait ~30 seconds for 3D reconstruction
4. **Check Godot logs**: Look for "New character received" messages

### "Pose Updates Not Working"
1. **Check pose confidence**: Low confidence poses are filtered out
2. **Check visibility**: Person must be fully visible (arms, face, torso)
3. **Check smoothing**: High smoothing causes lag

### Performance Issues
1. **Reduce avatar complexity**: Use simpler meshes
2. **Limit avatar count**: Remove old avatars when people leave
3. **Adjust pose update rate**: Lower FPS in Avatar Mirror config

## Configuration Options

In `AvatarMirrorClient.gd`:

```gdscript
# Connection settings
@export var server_host: String = "localhost"
@export var server_port: int = 8765
@export var auto_connect: bool = true
@export var reconnect_interval: float = 5.0

# Modify these in the Godot editor
```

In Avatar Mirror `config.json`:
```json
{
  "websocket": {
    "host": "localhost",
    "port": 8765,
    "max_clients": 10
  },
  "pose_estimation": {
    "fps": 30
  }
}
```

## Example Use Cases

### 1. Digital Mirror
- People see themselves as 3D avatars
- Real-time pose mirroring
- Multiple people simultaneously

### 2. Multiplayer Game
- Players enter room to create their avatar
- Avatar represents them in the game world
- Real-time body movement controls

### 3. Virtual Meeting Space  
- Participants appear as their real avatars
- Body language and poses preserved
- More immersive than video calls

### 4. Interactive Installation
- Museum or gallery installation
- Visitors become part of the experience
- Avatar interactions and effects

## Next Steps

1. **Test the basic connection** with the demo scene
2. **Create your custom Avatar scene** with your desired mesh/materials
3. **Integrate pose data** with your game/application logic
4. **Add visual effects** for avatar spawning, pose changes, etc.
5. **Optimize performance** for your specific use case

Happy avataring! 🎭✨