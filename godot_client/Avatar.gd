# Avatar Node for Avatar Mirror System
# Handles 3D rigged characters with pose updates
extends Node3D

class_name Avatar

# Avatar data
var person_id: int = -1
var character_data: Dictionary = {}
var skeleton: Skeleton3D
var mesh_instance: MeshInstance3D

# Pose data
var current_pose: Dictionary = {}
var pose_confidence: float = 0.0

# Joint mapping from MediaPipe to skeleton bones
var joint_mapping: Dictionary = {
	"nose": "Head",
	"left_eye": "LeftEye", 
	"right_eye": "RightEye",
	"left_shoulder": "LeftShoulder",
	"right_shoulder": "RightShoulder", 
	"left_elbow": "LeftElbow",
	"right_elbow": "RightElbow",
	"left_wrist": "LeftHand",
	"right_wrist": "RightHand",
	"left_hip": "LeftHip",
	"right_hip": "RightHip",
	"left_knee": "LeftKnee", 
	"right_knee": "RightKnee",
	"left_ankle": "LeftFoot",
	"right_ankle": "RightFoot"
}

# Pose smoothing
@export var pose_smoothing: float = 0.1
var target_transforms: Dictionary = {}

# Signals
signal pose_updated(pose_data: Dictionary)
signal model_loaded(success: bool)

func _ready():
	# Create default structure
	_setup_default_avatar()

func _setup_default_avatar():
	"""Setup basic avatar structure"""
	# Create mesh instance
	mesh_instance = MeshInstance3D.new()
	add_child(mesh_instance)
	
	# Create skeleton
	skeleton = Skeleton3D.new()
	add_child(skeleton)
	
	print("✅ Default avatar structure created")

func setup_character(char_data: Dictionary):
	"""Setup avatar with character data from Avatar Mirror"""
	character_data = char_data
	person_id = char_data.get("person_id", -1)
	
	print("🔧 Setting up character for person %d" % person_id)
	
	# Try to load the rigged model
	var mesh_path = char_data.get("mesh_path", "")
	if mesh_path != "":
		_load_rigged_model(mesh_path)
	else:
		_create_procedural_avatar()
	
	# Setup skeleton from character data
	var skeleton_data = char_data.get("skeleton_data", {})
	if not skeleton_data.is_empty():
		_setup_skeleton(skeleton_data)

func _load_rigged_model(mesh_path: String):
	"""Load rigged model from OBJ file"""
	print("📥 Loading rigged model: %s" % mesh_path)
	
	# Note: Godot doesn't directly support OBJ with rigging
	# You might need to convert to GLTF or use a plugin
	# For now, we'll create a procedural avatar
	_create_procedural_avatar()
	
	# In a real implementation, you would:
	# 1. Convert OBJ to GLTF with rigging data
	# 2. Load the GLTF scene
	# 3. Extract the mesh and skeleton
	
	emit_signal("model_loaded", true)

func _create_procedural_avatar():
	"""Create a procedural avatar mesh"""
	print("🎨 Creating procedural avatar")
	
	# Create a humanoid mesh procedurally
	var array_mesh = ArrayMesh.new()
	
	# Simple humanoid shape (you can make this more sophisticated)
	var vertices = PackedVector3Array()
	var normals = PackedVector3Array() 
	var uvs = PackedVector2Array()
	var indices = PackedInt32Array()
	
	# Head
	_add_sphere_to_mesh(vertices, normals, uvs, indices, Vector3(0, 1.7, 0), 0.15)
	
	# Torso  
	_add_cylinder_to_mesh(vertices, normals, uvs, indices, Vector3(0, 1.2, 0), Vector3(0, 0.8, 0), 0.2)
	
	# Arms
	_add_cylinder_to_mesh(vertices, normals, uvs, indices, Vector3(-0.25, 1.4, 0), Vector3(-0.6, 1.1, 0), 0.08)  # Left upper arm
	_add_cylinder_to_mesh(vertices, normals, uvs, indices, Vector3(-0.6, 1.1, 0), Vector3(-0.9, 0.8, 0), 0.06)   # Left forearm
	
	_add_cylinder_to_mesh(vertices, normals, uvs, indices, Vector3(0.25, 1.4, 0), Vector3(0.6, 1.1, 0), 0.08)   # Right upper arm
	_add_cylinder_to_mesh(vertices, normals, uvs, indices, Vector3(0.6, 1.1, 0), Vector3(0.9, 0.8, 0), 0.06)    # Right forearm
	
	# Legs
	_add_cylinder_to_mesh(vertices, normals, uvs, indices, Vector3(-0.1, 0.8, 0), Vector3(-0.1, 0.4, 0), 0.1)   # Left thigh
	_add_cylinder_to_mesh(vertices, normals, uvs, indices, Vector3(-0.1, 0.4, 0), Vector3(-0.1, 0.0, 0), 0.08)  # Left shin
	
	_add_cylinder_to_mesh(vertices, normals, uvs, indices, Vector3(0.1, 0.8, 0), Vector3(0.1, 0.4, 0), 0.1)     # Right thigh  
	_add_cylinder_to_mesh(vertices, normals, uvs, indices, Vector3(0.1, 0.4, 0), Vector3(0.1, 0.0, 0), 0.08)    # Right shin
	
	# Create mesh
	var arrays = []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_TEX_UV] = uvs
	arrays[Mesh.ARRAY_INDEX] = indices
	
	array_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	mesh_instance.mesh = array_mesh
	
	# Add material
	var material = StandardMaterial3D.new()
	material.albedo_color = Color(0.8, 0.7, 0.6)  # Skin tone
	mesh_instance.set_surface_override_material(0, material)

func _add_sphere_to_mesh(vertices: PackedVector3Array, normals: PackedVector3Array, 
						uvs: PackedVector2Array, indices: PackedInt32Array, 
						center: Vector3, radius: float):
	"""Add sphere geometry to mesh arrays"""
	var start_vertex = vertices.size()
	var rings = 8
	var sectors = 12
	
	for i in range(rings + 1):
		var lat = PI * (-0.5 + float(i) / rings)
		var y = sin(lat)
		var xz_radius = cos(lat)
		
		for j in range(sectors + 1):
			var lng = 2.0 * PI * float(j) / sectors
			var x = cos(lng) * xz_radius
			var z = sin(lng) * xz_radius
			
			var vertex = center + Vector3(x, y, z) * radius
			vertices.append(vertex)
			normals.append(Vector3(x, y, z))
			uvs.append(Vector2(float(j) / sectors, float(i) / rings))
	
	# Generate indices
	for i in range(rings):
		for j in range(sectors):
			var current = start_vertex + i * (sectors + 1) + j
			var next = current + sectors + 1
			
			indices.append(current)
			indices.append(next)
			indices.append(current + 1)
			
			indices.append(current + 1) 
			indices.append(next)
			indices.append(next + 1)

func _add_cylinder_to_mesh(vertices: PackedVector3Array, normals: PackedVector3Array,
						   uvs: PackedVector2Array, indices: PackedInt32Array,
						   start_pos: Vector3, end_pos: Vector3, radius: float):
	"""Add cylinder geometry to mesh arrays"""
	var start_vertex = vertices.size()
	var segments = 8
	var direction = (end_pos - start_pos).normalized()
	var length = start_pos.distance_to(end_pos)
	
	# Create local coordinate system
	var up = Vector3.UP
	if abs(direction.dot(up)) > 0.9:
		up = Vector3.RIGHT
	var right = direction.cross(up).normalized()
	up = right.cross(direction).normalized()
	
	# Add vertices
	for segment in range(segments + 1):
		var angle = 2.0 * PI * segment / segments
		var x = cos(angle)
		var z = sin(angle)
		
		var offset = (right * x + up * z) * radius
		
		# Bottom vertex
		vertices.append(start_pos + offset)
		normals.append(offset.normalized())
		uvs.append(Vector2(float(segment) / segments, 0.0))
		
		# Top vertex
		vertices.append(end_pos + offset)
		normals.append(offset.normalized()) 
		uvs.append(Vector2(float(segment) / segments, 1.0))
	
	# Generate indices
	for segment in range(segments):
		var current_bottom = start_vertex + segment * 2
		var current_top = current_bottom + 1
		var next_bottom = start_vertex + ((segment + 1) % (segments + 1)) * 2
		var next_top = next_bottom + 1
		
		# Side faces
		indices.append(current_bottom)
		indices.append(next_bottom) 
		indices.append(current_top)
		
		indices.append(current_top)
		indices.append(next_bottom)
		indices.append(next_top)

func _setup_skeleton(skeleton_data: Dictionary):
	"""Setup skeleton from Avatar Mirror skeleton data"""
	print("🦴 Setting up skeleton")
	
	var joint_positions = skeleton_data.get("joint_positions", {})
	var bone_hierarchy = skeleton_data.get("bone_hierarchy", {})
	
	# Clear existing bones
	skeleton.clear_bones()
	
	# Add bones based on joint positions
	for joint_name in joint_positions:
		var bone_id = skeleton.get_bone_count()
		skeleton.add_bone(joint_name)
		
		var position = joint_positions[joint_name]
		if typeof(position) == TYPE_ARRAY and position.size() >= 3:
			var rest_transform = Transform3D()
			rest_transform.origin = Vector3(position[0], position[1], position[2])
			skeleton.set_bone_rest(bone_id, rest_transform)

func update_pose(pose_data: Dictionary):
	"""Update avatar pose from Avatar Mirror pose data"""
	current_pose = pose_data
	pose_confidence = pose_data.get("confidence", 0.0)
	
	var joints = pose_data.get("joints", {})
	
	# Update transforms for each joint
	for joint_name in joints:
		if joint_mapping.has(joint_name):
			var bone_name = joint_mapping[joint_name]
			var joint_data = joints[joint_name]
			var position = joint_data.get("position", [0, 0, 0])
			var confidence = joint_data.get("confidence", 0.0)
			
			if confidence > 0.3:  # Only update if confidence is high enough
				_update_bone_transform(bone_name, position)
	
	emit_signal("pose_updated", pose_data)

func _update_bone_transform(bone_name: String, position: Array):
	"""Update bone transform with smoothing"""
	if skeleton.get_bone_count() == 0:
		return
	
	var bone_id = skeleton.find_bone(bone_name)
	if bone_id == -1:
		return
	
	var target_pos = Vector3(position[0], position[1], position[2])
	
	# Apply smoothing
	if target_transforms.has(bone_name):
		var current_pos = target_transforms[bone_name]
		target_pos = current_pos.lerp(target_pos, pose_smoothing)
	
	target_transforms[bone_name] = target_pos
	
	# Update bone transform
	var transform = Transform3D()
	transform.origin = target_pos
	skeleton.set_bone_global_pose_override(bone_id, transform, 1.0, true)

func get_pose_confidence() -> float:
	"""Get current pose confidence"""
	return pose_confidence

func get_joint_position(joint_name: String) -> Vector3:
	"""Get position of specific joint"""
	var joints = current_pose.get("joints", {})
	if joints.has(joint_name):
		var pos = joints[joint_name].get("position", [0, 0, 0])
		return Vector3(pos[0], pos[1], pos[2])
	return Vector3.ZERO

func set_pose_smoothing(smoothing: float):
	"""Set pose smoothing factor (0-1, higher = more smoothing)"""
	pose_smoothing = clamp(smoothing, 0.0, 1.0)