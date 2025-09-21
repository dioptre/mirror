import asyncio
import websockets
import json
import logging
from typing import Dict, Any, List, Optional, Callable
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class GodotWebSocketServer:
    """WebSocket server for communicating with Godot"""
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        self.running = False
        
        # Message queues
        self.character_queue = asyncio.Queue()
        self.pose_queue = asyncio.Queue()
        
        # Callbacks
        self.message_handlers = {}
        
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        
        self.running = True
        logger.info("WebSocket server started successfully")
        
        # Start message processing task
        asyncio.create_task(self.process_messages())
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        logger.info("WebSocket server stopped")
    
    async def handle_client(self, websocket):
        """Handle new client connection"""
        client_address = websocket.remote_address
        logger.info(f"New client connected: {client_address}")
        
        self.clients.add(websocket)
        
        try:
            # Send welcome message
            welcome_msg = {
                'type': 'welcome',
                'message': 'Connected to Avatar Mirror system',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Listen for messages from client
            async for message in websocket:
                await self.handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_address}")
        except Exception as e:
            logger.error(f"Error handling client {client_address}: {e}")
        finally:
            self.clients.discard(websocket)
    
    async def handle_message(self, websocket, raw_message: str):
        """Handle incoming message from client"""
        try:
            message = json.loads(raw_message)
            message_type = message.get('type')
            
            logger.debug(f"Received message type: {message_type}")
            
            # Handle different message types
            if message_type in self.message_handlers:
                response = await self.message_handlers[message_type](message)
                if response:
                    await websocket.send(json.dumps(response))
            else:
                # Default response for unknown message types
                response = {
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }
                await websocket.send(json.dumps(response))
                
        except json.JSONDecodeError:
            error_response = {
                'type': 'error',
                'message': 'Invalid JSON format'
            }
            await websocket.send(json.dumps(error_response))
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message type"""
        self.message_handlers[message_type] = handler
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.clients:
            return
        
        message_str = json.dumps(message)
        disconnected_clients = []
        
        for client in self.clients:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.discard(client)
    
    async def send_character_data(self, character_data: Dict[str, Any]):
        """Send new character data to Godot"""
        message = {
            'type': 'new_character',
            'data': character_data,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.character_queue.put(message)
    
    async def send_pose_data(self, pose_data: Dict[int, Dict[str, Any]]):
        """Send pose data to Godot"""
        message = {
            'type': 'pose_update',
            'data': pose_data,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.pose_queue.put(message)
    
    async def process_messages(self):
        """Process queued messages"""
        while self.running:
            try:
                # Process character messages
                try:
                    character_msg = self.character_queue.get_nowait()
                    await self.broadcast_message(character_msg)
                except asyncio.QueueEmpty:
                    pass
                
                # Process pose messages
                try:
                    pose_msg = self.pose_queue.get_nowait()
                    await self.broadcast_message(pose_msg)
                except asyncio.QueueEmpty:
                    pass
                
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
    
    def get_client_count(self) -> int:
        """Get number of connected clients"""
        return len(self.clients)


class GodotCommunicator:
    """High-level communicator for Godot integration"""
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.server = GodotWebSocketServer(host, port)
        self.thread = None
        self.loop = None
        
        # Setup message handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup message handlers"""
        self.server.register_message_handler('ping', self._handle_ping)
        self.server.register_message_handler('client_info', self._handle_client_info)
        self.server.register_message_handler('get_characters', self._handle_get_characters)
        self.server.register_message_handler('get_poses', self._handle_get_poses)
    
    async def _handle_ping(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping message"""
        return {
            'type': 'pong',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_client_info(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle client info message"""
        client_data = message.get('data', {})
        client_type = client_data.get('client_type', 'unknown')
        
        logger.info(f"Client connected: {client_type} v{client_data.get('version', 'unknown')}")
        
        return {
            'type': 'client_info_ack',
            'data': {
                'server_type': 'avatar_mirror',
                'version': '1.0.0',
                'capabilities': ['face_detection', 'pose_estimation', '3d_reconstruction', 'face_swapping']
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_get_characters(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle request for character list"""
        # This would be implemented based on your caching system
        return {
            'type': 'character_list',
            'characters': [],  # Placeholder
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_get_poses(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle request for current poses"""
        # This would return current pose data
        return {
            'type': 'current_poses',
            'poses': {},  # Placeholder
            'timestamp': datetime.now().isoformat()
        }
    
    def start(self):
        """Start the WebSocket server in a separate thread"""
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            try:
                self.loop.run_until_complete(self.server.start_server())
                self.loop.run_forever()
            except Exception as e:
                logger.error(f"WebSocket server error: {e}")
            finally:
                self.loop.close()
        
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        
        # Wait a moment for server to start
        time.sleep(0.5)
        logger.info("Godot communicator started")
    
    def stop(self):
        """Stop the WebSocket server"""
        if self.loop and self.loop.is_running():
            # Schedule the stop coroutine
            asyncio.run_coroutine_threadsafe(self.server.stop_server(), self.loop)
            
            # Stop the event loop
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        logger.info("Godot communicator stopped")
    
    def notify_new_character(self, character_data: Dict[str, Any]):
        """Notify Godot of a new character"""
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.server.send_character_data(character_data),
                self.loop
            )
    
    def update_poses(self, pose_data: Dict[int, Dict[str, Any]]):
        """Update poses in Godot"""
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.server.send_pose_data(pose_data),
                self.loop
            )
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self.server.running
    
    def get_client_count(self) -> int:
        """Get number of connected Godot clients"""
        return self.server.get_client_count()


class CharacterNotificationSystem:
    """System for notifying Godot about new characters and models"""
    
    def __init__(self, communicator: GodotCommunicator):
        self.communicator = communicator
        self.notified_characters = set()
    
    def notify_new_character(self, person_id: int, rig_result: Dict[str, Any]):
        """Notify Godot about a new rigged character"""
        if person_id in self.notified_characters:
            return  # Already notified
        
        if not rig_result.get('success', False):
            return  # Rigging failed
        
        character_data = {
            'person_id': person_id,
            'mesh_path': rig_result.get('rig_path'),
            'skeleton_data': rig_result.get('skeleton'),
            'joint_names': rig_result.get('joint_names'),
            'timestamp': datetime.now().isoformat()
        }
        
        self.communicator.notify_new_character(character_data)
        self.notified_characters.add(person_id)
        
        logger.info(f"Notified Godot about new character: person {person_id}")
    
    def remove_character(self, person_id: int):
        """Remove character from notification tracking"""
        self.notified_characters.discard(person_id)


class PoseStreamingSystem:
    """System for streaming pose data to Godot"""
    
    def __init__(self, communicator: GodotCommunicator, fps: int = 30):
        self.communicator = communicator
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.last_update = 0
        self.latest_poses = {}
    
    def update_poses(self, pose_data: Dict[int, Dict[str, Any]]):
        """Update pose data (called from main loop)"""
        current_time = time.time()
        
        # Throttle updates to target FPS
        if current_time - self.last_update < self.frame_time:
            return
        
        self.latest_poses = pose_data
        self.communicator.update_poses(pose_data)
        self.last_update = current_time
    
    def set_fps(self, fps: int):
        """Set target FPS for pose streaming"""
        self.fps = fps
        self.frame_time = 1.0 / fps


class GodotIntegration:
    """Complete Godot integration system"""
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.communicator = GodotCommunicator(host, port)
        self.character_system = CharacterNotificationSystem(self.communicator)
        self.pose_system = PoseStreamingSystem(self.communicator)
        
    def start(self):
        """Start Godot integration"""
        self.communicator.start()
    
    def stop(self):
        """Stop Godot integration"""
        self.communicator.stop()
    
    def notify_new_character(self, person_id: int, rig_result: Dict[str, Any]):
        """Notify about new character"""
        self.character_system.notify_new_character(person_id, rig_result)
    
    def update_poses(self, pose_data: Dict[int, Dict[str, Any]]):
        """Update pose data"""
        self.pose_system.update_poses(pose_data)
    
    def is_connected(self) -> bool:
        """Check if Godot clients are connected"""
        return self.communicator.get_client_count() > 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            'server_running': self.communicator.is_running(),
            'client_count': self.communicator.get_client_count(),
            'characters_notified': len(self.character_system.notified_characters)
        }