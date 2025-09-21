# 🚀 Quick Start Guide - Avatar Mirror System

Get your Avatar Mirror working in **3 simple steps**!

## Step 1: Setup the System ⚙️

```bash
# Navigate to the project
cd /Users/andrewgrosser/Documents/mirror

# Download all AI models (one-time setup)
make setup-models

# This downloads ~284MB of AI models and repositories
```

## Step 2: Start Avatar Mirror 🎥

```bash
# Start the Avatar Mirror System
make run
```

You should see:
```
✅ All components initialized successfully
🎥 Camera initialized: 1920x1080 @ 30.0fps
📡 WebSocket server started on localhost:8765
```

## Step 3: Connect Godot 🎮

### Option A: Auto Demo
```bash
# In a new terminal (keep Avatar Mirror running)
make demo
```

### Option B: Manual
```bash
# Open Godot project
make godot

# In Godot:
# 1. Press F5 to run
# 2. Select MirrorDemo.tscn as main scene
# 3. Click "Select Current"
```

## 🎬 What Happens Next

1. **Stand in front of your camera**
2. **Avatar Mirror detects you** (see "Person X detected" in logs)
3. **Wait ~30 seconds** for 3D avatar creation
4. **Your avatar appears in Godot** with real-time pose tracking!

## 🎮 Demo Controls

- **Arrow Keys**: Orbit camera around avatars
- **R**: Request character list
- **P**: Request current poses
- **C**: Clear all avatars  
- **Space**: Toggle recording mode
- **ESC**: Quit

## 🔧 Troubleshooting

### "Camera not detected"
```bash
# Check available cameras
ls /dev/video*  # Linux
# Or check camera permissions on macOS
```

### "Connection failed"
```bash
# Check Avatar Mirror is running
ps aux | grep python

# Check WebSocket port
netstat -an | grep 8765
```

### "No avatars appearing"
1. **Ensure good lighting** - person should be well-lit
2. **Full body visible** - arms, face, and torso must be visible
3. **Wait for processing** - 3D reconstruction takes ~30 seconds
4. **Check logs** - look for "Person X confirmed" messages

## ⚡ Performance Tips

- **Lower camera resolution** for faster processing
- **Good lighting** improves detection quality
- **Stable pose** gets better 3D reconstruction
- **CPU usage** is normal during 3D processing

## 🎯 Success Indicators

✅ **Avatar Mirror System**: "WebSocket server started"
✅ **Godot Connection**: "Connected to Avatar Mirror System!" 
✅ **Person Detection**: "Person X detected"
✅ **Avatar Creation**: "New character received: Person X"
✅ **Pose Tracking**: Real-time avatar movement

Your **digital mirror** is now live! 🪞✨

## Next Steps

1. **Customize avatars** - Edit `Avatar.gd` for custom meshes
2. **Add effects** - Create spawn/despawn animations  
3. **Multiple cameras** - Support different viewing angles
4. **Recording** - Save pose sequences for playback
5. **Multiplayer** - Connect multiple Godot clients

**Happy avataring!** 🎭🤖