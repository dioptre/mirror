# Avatar Mirror Client for Godot
# Connects to the Avatar Mirror System via WebSocket
extends Node

class_name AvatarMirrorClient

# WebSocket connection
var websocket_peer: WebSocketPeer
var is_connected: bool = false

# Avatar Mirror settings
@export var server_host: String = "localhost"
@export var server_port: int = 8765
@export var auto_connect: bool = true
@export var reconnect_interval: float = 5.0

# Avatar management
var avatars: Dictionary = {}  # person_id -> Avatar node
var avatar_scene: PackedScene  # Prefab for avatars

# Signals
signal connected_to_server()
signal disconnected_from_server() 
signal character_received(character_data: Dictionary)
signal pose_updated(person_id: int, pose_data: Dictionary)
signal connection_error(error: String)

func _ready():
	print("🚀 Avatar Mirror Client starting...")
	
	# Initialize WebSocket
	websocket_peer = WebSocketPeer.new()
	
	# Load avatar scene
	if ResourceLoader.exists("res://Avatar.tscn"):
		avatar_scene = load("res://Avatar.tscn")
		print("✅ Avatar scene loaded")
	else:
		print("⚠️ Avatar scene not found, will use simple avatars")
	
	if auto_connect:
		connect_to_server()

func _process(_delta):
	if websocket_peer.get_ready_state() != WebSocketPeer.STATE_CLOSED:
		websocket_peer.poll()
		
		# Handle connection state
		var state = websocket_peer.get_ready_state()
		if state == WebSocketPeer.STATE_OPEN and not is_connected:
			_on_connection_established()
		elif state == WebSocketPeer.STATE_CLOSED and is_connected:
			_on_connection_closed()
		
		# Process incoming messages
		while websocket_peer.get_available_packet_count():
			_process_message()

func connect_to_server():
	var url = "ws://%s:%d" % [server_host, server_port]
	print("🔌 Connecting to Avatar Mirror at %s..." % url)
	
	var error = websocket_peer.connect_to_url(url)
	if error != OK:
		print("❌ Failed to connect: %s" % error)
		emit_signal("connection_error", "Failed to connect to server")

func disconnect_from_server():
	if is_connected:
		websocket_peer.close(1000, "Client disconnect")
		is_connected = false

func _on_connection_established():
	is_connected = true
	print("✅ Connected to Avatar Mirror System!")
	emit_signal("connected_to_server")
	
	# Send initial handshake
	_send_message({
		"type": "client_info",
		"data": {
			"client_type": "godot",
			"version": "1.0",
			"capabilities": ["avatars", "poses", "real_time"]
		}
	})

func _on_connection_closed():
	is_connected = false
	print("🔌 Disconnected from Avatar Mirror System")
	emit_signal("disconnected_from_server")
	
	# Auto-reconnect
	if auto_connect:
		print("🔄 Reconnecting in %s seconds..." % reconnect_interval)
		await get_tree().create_timer(reconnect_interval).timeout
		connect_to_server()

func _process_message():
	var packet = websocket_peer.get_packet()
	var message_text = packet.get_string_from_utf8()
	
	var json = JSON.new()
	var parse_result = json.parse(message_text)
	
	if parse_result != OK:
		print("❌ Failed to parse message: %s" % message_text)
		return
	
	var message = json.data
	_handle_message(message)

func _handle_message(message: Dictionary):
	var message_type = message.get("type", "")
	
	match message_type:
		"welcome":
			print("👋 %s" % message.get("message", "Welcome"))
		
		"client_info_ack":
			print("🤝 Server acknowledged client info")
			var server_data = message.get("data", {})
			var capabilities = server_data.get("capabilities", [])
			print("   Server capabilities: " + str(capabilities))
		
		"new_character":
			_handle_new_character(message.get("data", {}))
		
		"pose_update":
			_handle_pose_update(message.get("data", {}))
		
		"character_list":
			var characters = message.get("characters", [])
			print("📋 Character list received: %d characters" % characters.size())
		
		"current_poses":
			var poses = message.get("poses", {})
			print("🤸 Current poses received for %d people" % poses.size())
		
		"error":
			print("❌ Server error: %s" % message.get("message", "Unknown error"))
			emit_signal("connection_error", message.get("message", "Unknown error"))
		
		_:
			print("🤷 Unknown message type: %s" % message_type)

func _handle_new_character(character_data: Dictionary):
	var person_id = character_data.get("person_id", -1)
	if person_id == -1:
		print("❌ Invalid character data: missing person_id")
		return
	
	print("🧑 New character received: Person %d" % person_id)
	
	# Create new avatar
	_create_avatar(person_id, character_data)
	
	emit_signal("character_received", character_data)

func _handle_pose_update(pose_data: Dictionary):
	for person_id_str in pose_data:
		var person_id = int(person_id_str)
		var pose = pose_data[person_id_str]
		
		if avatars.has(person_id):
			_update_avatar_pose(person_id, pose)
		
		emit_signal("pose_updated", person_id, pose)

func _create_avatar(person_id: int, character_data: Dictionary):
	if avatars.has(person_id):
		print("⚠️ Avatar for person %d already exists, removing old one" % person_id)
		_remove_avatar(person_id)
	
	var avatar_node: Node3D
	
	if avatar_scene:
		avatar_node = avatar_scene.instantiate()
	else:
		# Fallback: create simple avatar
		avatar_node = _create_simple_avatar(person_id)
	
	if avatar_node:
		avatar_node.name = "Avatar_%d" % person_id
		add_child(avatar_node)
		avatars[person_id] = avatar_node
		
		# Setup avatar with character data
		if avatar_node.has_method("setup_character"):
			avatar_node.setup_character(character_data)
		
		print("✅ Avatar created for person %d" % person_id)

func _create_simple_avatar(person_id: int) -> Node3D:
	"""Create a simple avatar as fallback"""
	var avatar = Node3D.new()
	
	# Create a simple capsule body
	var mesh_instance = MeshInstance3D.new()
	var capsule_mesh = CapsuleMesh.new()
	capsule_mesh.height = 1.8
	capsule_mesh.top_radius = 0.3
	capsule_mesh.bottom_radius = 0.3
	mesh_instance.mesh = capsule_mesh
	
	# Add material
	var material = StandardMaterial3D.new()
	material.albedo_color = Color(randf(), randf(), randf())  # Random color per person
	mesh_instance.set_surface_override_material(0, material)
	
	avatar.add_child(mesh_instance)
	
	# Add a label
	var label_3d = Label3D.new()
	label_3d.text = "Person %d" % person_id
	label_3d.position = Vector3(0, 1.0, 0)
	avatar.add_child(label_3d)
	
	return avatar

func _update_avatar_pose(person_id: int, pose_data: Dictionary):
	var avatar = avatars.get(person_id)
	if not avatar:
		return
	
	if not pose_data.get("pose_present", false):
		return
	
	# Update avatar pose
	if avatar.has_method("update_pose"):
		avatar.update_pose(pose_data)
	else:
		_update_simple_avatar_pose(avatar, pose_data)

func _update_simple_avatar_pose(avatar: Node3D, pose_data: Dictionary):
	"""Update simple avatar pose based on MediaPipe landmarks"""
	var joints = pose_data.get("joints", {})
	
	# Simple pose update - you can expand this
	if joints.has("nose") and joints.has("left_shoulder") and joints.has("right_shoulder"):
		var nose_pos = joints["nose"]["position"]
		var left_shoulder = joints["left_shoulder"]["position"]
		var right_shoulder = joints["right_shoulder"]["position"]
		
		# Calculate head position and body orientation
		var head_pos = Vector3(nose_pos[0], nose_pos[1], nose_pos[2])
		var left_pos = Vector3(left_shoulder[0], left_shoulder[1], left_shoulder[2])
		var right_pos = Vector3(right_shoulder[0], right_shoulder[1], right_shoulder[2])
		
		# Update avatar position (simple mapping)
		avatar.position = head_pos * 2.0  # Scale factor
		
		# Calculate rotation from shoulder line
		var shoulder_dir = (right_pos - left_pos).normalized()
		if shoulder_dir.length() > 0:
			var forward = Vector3(0, 0, -1)
			var right = shoulder_dir
			var up = right.cross(forward).normalized()
			avatar.basis = Basis(right, up, forward)

func _remove_avatar(person_id: int):
	if avatars.has(person_id):
		var avatar = avatars[person_id]
		avatar.queue_free()
		avatars.erase(person_id)
		print("🗑️ Removed avatar for person %d" % person_id)

func _send_message(message: Dictionary):
	if not is_connected:
		return
	
	var json_string = JSON.stringify(message)
	websocket_peer.send_text(json_string)

func request_character_list():
	"""Request list of current characters"""
	_send_message({"type": "get_characters"})

func request_current_poses():
	"""Request current pose data"""
	_send_message({"type": "get_poses"})

func get_avatar(person_id: int) -> Node3D:
	"""Get avatar node for person ID"""
	return avatars.get(person_id)

func get_avatar_count() -> int:
	"""Get number of active avatars"""
	return avatars.size()

func clear_all_avatars():
	"""Remove all avatars"""
	for person_id in avatars.keys():
		_remove_avatar(person_id)
