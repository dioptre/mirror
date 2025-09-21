# Godot Client for Avatar Mirror System

This directory contains a complete Godot client implementation for the Avatar Mirror System.

## Files

- **`AvatarMirrorClient.gd`** - Main WebSocket client that connects to the Avatar Mirror System
- **`Avatar.gd`** - Avatar node that represents a person with pose updates
- **`MirrorDemo.gd`** - Demo scene showing how to use the system
- **`project.godot`** - Godot project configuration

## Quick Setup

1. **Open in Godot**:
   - Open Godot 4.3+
   - Import this directory as a project
   - The main scene is `MirrorDemo.tscn`

2. **Start Avatar Mirror System**:
   ```bash
   # In the main avatar-mirror directory
   make run
   ```

3. **Run the Godot demo**:
   - Press F5 in Godot
   - The client will automatically connect to `ws://localhost:8765`

## Usage

### Basic Connection

```gdscript
# Add AvatarMirrorClient to your scene
var mirror_client = AvatarMirrorClient.new()
add_child(mirror_client)

# Connect signals
mirror_client.character_received.connect(_on_new_character)
mirror_client.pose_updated.connect(_on_pose_update)

func _on_new_character(character_data: Dictionary):
    print("New person detected: ", character_data.person_id)

func _on_pose_update(person_id: int, pose_data: Dictionary):
    print("Pose update for person ", person_id)
```

### Advanced Avatar Handling

```gdscript
# Get specific avatar
var avatar = mirror_client.get_avatar(person_id)
if avatar:
    var confidence = avatar.get_pose_confidence()
    var nose_pos = avatar.get_joint_position("nose")

# Configure pose smoothing
avatar.set_pose_smoothing(0.2)  # Higher = smoother but more lag
```

## WebSocket Protocol

The client handles these message types automatically:

### Incoming Messages

- **`welcome`** - Server connection confirmation
- **`new_character`** - New person detected and processed
- **`pose_update`** - Real-time pose data for all people
- **`error`** - Error messages from server

### Outgoing Messages

- **`client_info`** - Client capabilities (sent on connect)
- **`get_characters`** - Request current character list
- **`get_poses`** - Request current pose data

## Controls (in demo)

- **Arrow keys** - Orbit camera around avatars
- **R** - Request character list from server
- **P** - Request current poses
- **C** - Clear all avatars
- **Space** - Toggle recording mode
- **ESC** - Quit

## Customization

### Creating Custom Avatars

1. Create your own Avatar scene inheriting from the Avatar class:
```gdscript
extends Avatar

func _ready():
    super._ready()
    # Your custom setup
```

2. Override pose handling:
```gdscript
func update_pose(pose_data: Dictionary):
    super.update_pose(pose_data)
    # Your custom pose processing
```

### Adding Visual Effects

```gdscript
# Connect to avatar events
avatar_mirror_client.character_received.connect(_create_spawn_effect)

func _create_spawn_effect(character_data):
    # Create particle effects, sounds, etc.
```

### Custom Materials and Shading

The procedural avatars use StandardMaterial3D. You can:

1. Replace with custom materials
2. Add texture mapping based on person data
3. Implement different avatar styles per person

## Integration with Existing Projects

To integrate into your existing Godot project:

1. Copy `AvatarMirrorClient.gd` and `Avatar.gd` to your project
2. Add an AvatarMirrorClient node to your main scene
3. Connect the signals to your existing avatar/character system
4. Map the pose data to your character rigs

## Troubleshooting

### Connection Issues

- Ensure Avatar Mirror System is running (`make run`)
- Check that port 8765 is not blocked by firewall
- Verify the server host/port in the client settings

### Performance Issues

- Reduce pose smoothing for more responsive avatars
- Limit the number of avatars if performance drops
- Use simpler avatar meshes for better performance

### Avatar Display Issues

- Check that the Avatar Mirror System is detecting people properly
- Ensure good lighting and full body visibility for 3D reconstruction
- Verify that pose confidence is above 0.3 for stable tracking

## Advanced Features

### Multiple Camera Views

```gdscript
# Add multiple cameras for different angles
var front_camera = Camera3D.new()
var side_camera = Camera3D.new()
var top_camera = Camera3D.new()

# Switch between views based on avatar activity
```

### Avatar Interaction

```gdscript
# Detect when avatars are close to each other
func _check_avatar_proximity():
    for person_id in avatars:
        var avatar = avatars[person_id]
        # Check distances, trigger interactions
```

### Recording and Playback

```gdscript
# Record pose sequences
var pose_recording = []

func _on_pose_updated(person_id, pose_data):
    if is_recording:
        pose_recording.append({
            "person_id": person_id,
            "pose": pose_data,
            "timestamp": Time.get_time_dict_from_system()
        })
```

## Next Steps

1. **Create your own Avatar scene** with custom meshes and materials
2. **Integrate with your game/application** using the WebSocket client
3. **Add visual effects** for avatar spawning, pose changes, etc.
4. **Implement avatar persistence** to remember people between sessions
5. **Add multiplayer features** if running multiple Godot clients