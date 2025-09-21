# Avatar Mirror Demo Scene
# Demonstrates the Avatar Mirror System in Godot
extends Node3D

@export var camera_distance: float = 5.0
@export var camera_height: float = 2.0
@export var enable_debug_ui: bool = true

# Nodes
var avatar_mirror_client: AvatarMirrorClient
var camera_3d: Camera3D
var ui_layer: CanvasLayer
var debug_label: Label

# Demo state
var is_recording: bool = false
var avatar_count: int = 0

func _ready():
	print("🪞 Avatar Mirror Demo starting...")
	
	# Setup 3D scene
	_setup_scene()
	
	# Create Avatar Mirror client
	avatar_mirror_client = AvatarMirrorClient.new()
	add_child(avatar_mirror_client)
	
	# Connect signals
	avatar_mirror_client.connected_to_server.connect(_on_connected)
	avatar_mirror_client.disconnected_from_server.connect(_on_disconnected)
	avatar_mirror_client.character_received.connect(_on_character_received)
	avatar_mirror_client.pose_updated.connect(_on_pose_updated)
	avatar_mirror_client.connection_error.connect(_on_connection_error)
	
	# Setup UI
	if enable_debug_ui:
		_setup_debug_ui()
	
	print("✅ Avatar Mirror Demo ready!")

func _setup_scene():
	"""Setup the 3D scene"""
	# Create camera
	camera_3d = Camera3D.new()
	camera_3d.position = Vector3(0, camera_height, camera_distance)
	camera_3d.look_at(Vector3.ZERO, Vector3.UP)
	add_child(camera_3d)
	
	# Add some lighting
	var directional_light = DirectionalLight3D.new()
	directional_light.position = Vector3(2, 3, 2)
	directional_light.look_at(Vector3.ZERO, Vector3.UP)
	add_child(directional_light)
	
	# Add ground plane
	var ground = MeshInstance3D.new()
	var plane_mesh = PlaneMesh.new()
	plane_mesh.size = Vector2(10, 10)
	ground.mesh = plane_mesh
	
	var ground_material = StandardMaterial3D.new()
	ground_material.albedo_color = Color(0.3, 0.3, 0.3)
	ground.set_surface_override_material(0, ground_material)
	
	add_child(ground)
	
	print("🏟️ 3D scene setup complete")

func _setup_debug_ui():
	"""Setup debug UI overlay"""
	ui_layer = CanvasLayer.new()
	add_child(ui_layer)
	
	# Debug info panel
	var panel = Panel.new()
	panel.set_anchors_and_offsets_preset(Control.PRESET_TOP_LEFT)
	panel.size = Vector2(400, 300)
	panel.position = Vector2(20, 20)
	ui_layer.add_child(panel)
	
	var vbox = VBoxContainer.new()
	vbox.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	vbox.add_theme_constant_override("separation", 10)
	panel.add_child(vbox)
	
	# Title
	var title = Label.new()
	title.text = "🪞 Avatar Mirror System"
	title.add_theme_font_size_override("font_size", 24)
	vbox.add_child(title)
	
	# Status
	debug_label = Label.new()
	debug_label.text = "Status: Starting..."
	debug_label.autowrap_mode = TextServer.AUTOWRAP_WORD
	vbox.add_child(debug_label)
	
	# Controls
	var controls_label = Label.new()
	controls_label.text = "Controls:\n• R - Request character list\n• P - Request poses\n• C - Clear avatars\n• ESC - Quit"
	vbox.add_child(controls_label)
	
	print("🎛️ Debug UI setup complete")

func _input(event):
	"""Handle input events"""
	if event.is_action_pressed("ui_cancel"):
		get_tree().quit()
	
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_R:
				if avatar_mirror_client.is_connected:
					avatar_mirror_client.request_character_list()
					print("📋 Requested character list")
			
			KEY_P:
				if avatar_mirror_client.is_connected:
					avatar_mirror_client.request_current_poses()
					print("🤸 Requested current poses")
			
			KEY_C:
				avatar_mirror_client.clear_all_avatars()
				avatar_count = 0
				print("🗑️ Cleared all avatars")
			
			KEY_SPACE:
				is_recording = !is_recording
				print("🎥 Recording: %s" % ("ON" if is_recording else "OFF"))

func _process(_delta):
	"""Update demo state"""
	if debug_label:
		_update_debug_info()
	
	# Simple camera orbit
	if Input.is_action_pressed("ui_right"):
		rotate_y(-1.0 * _delta)
	if Input.is_action_pressed("ui_left"):
		rotate_y(1.0 * _delta)
	
	# Update camera position
	if camera_3d:
		var orbit_position = Vector3(
			sin(rotation.y) * camera_distance,
			camera_height,
			cos(rotation.y) * camera_distance
		)
		camera_3d.position = orbit_position
		camera_3d.look_at(Vector3.ZERO, Vector3.UP)

func _update_debug_info():
	"""Update debug information display"""
	var status_text = ""
	
	if avatar_mirror_client.is_connected:
		status_text = "✅ Connected to Avatar Mirror"
	else:
		status_text = "🔌 Connecting..."
	
	status_text += "\n\n"
	status_text += "Avatars: %d\n" % avatar_count
	status_text += "Recording: %s\n" % ("🔴 ON" if is_recording else "⚪ OFF")
	
	if avatar_count > 0:
		status_text += "\nActive People:\n"
		for person_id in avatar_mirror_client.avatars:
			var avatar = avatar_mirror_client.get_avatar(person_id)
			if avatar and avatar.has_method("get_pose_confidence"):
				var confidence = avatar.get_pose_confidence()
				status_text += "• Person %d (confidence: %.2f)\n" % [person_id, confidence]
			else:
				status_text += "• Person %d\n" % person_id
	
	debug_label.text = status_text

# Signal handlers
func _on_connected():
	print("🎉 Connected to Avatar Mirror System!")
	if debug_label:
		debug_label.text = "✅ Connected!"

func _on_disconnected():
	print("📡 Disconnected from Avatar Mirror System")

func _on_character_received(character_data: Dictionary):
	var person_id = character_data.get("person_id", -1)
	avatar_count += 1
	print("🧑 New character: Person %d (Total: %d)" % [person_id, avatar_count])
	
	# You could add visual effects here
	_create_spawn_effect(Vector3(randf_range(-2, 2), 0, randf_range(-2, 2)))

func _on_pose_updated(person_id: int, pose_data: Dictionary):
	# Pose updates are handled automatically by avatars
	# You could add additional effects here
	pass

func _on_connection_error(error: String):
	print("❌ Connection error: %s" % error)

func _create_spawn_effect(position: Vector3):
	"""Create a visual effect when a new avatar spawns"""
	var effect = MeshInstance3D.new()
	var sphere_mesh = SphereMesh.new()
	sphere_mesh.radius = 0.1
	effect.mesh = sphere_mesh
	
	var material = StandardMaterial3D.new()
	material.albedo_color = Color.GREEN
	material.emission = Color.GREEN * 0.5
	effect.set_surface_override_material(0, material)
	
	effect.position = position
	add_child(effect)
	
	# Animate the effect
	var tween = create_tween()
	tween.parallel().tween_property(effect, "scale", Vector3.ZERO, 1.0)
	tween.parallel().tween_property(effect, "position:y", position.y + 2, 1.0)
	tween.tween_callback(effect.queue_free)
