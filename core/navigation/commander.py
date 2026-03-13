import socket
import json
import threading
import time
import numpy as np
from . import utils


class CommandSender:
    def __init__(self, unity_host='127.0.0.1', command_port=5002):
        self.unity_host = unity_host
        self.command_port = command_port
        self.sock = None
        self.connected = False
        self.lock = threading.Lock()
        self._connect()

    def _connect(self):
        """Establish connection to Unity's command server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.unity_host, self.command_port))
            self.connected = True
            print(f"Connected to Unity command server on {self.unity_host}:{self.command_port}")
        except Exception as e:
            print(f"Failed to connect to Unity command server: {e}")
            self.connected = False
            self.sock = None

    def send_goal(self, x, y, theta=0.0):
        """Send a move_to command to Unity."""
        if not self.connected:
            print("Not connected to Unity, cannot send goal")
            return False
        msg = json.dumps({
            "command": "move_to",
            "x": float(x),
            "y": float(y),
            "theta": float(theta)
        }) + "\n"
        try:
            with self.lock:
                self.sock.sendall(msg.encode('utf-8'))
                print(f"Sent goal: ({x:.2f}, {y:.2f}, {theta:.2f})")
            return True
        except Exception as e:
            print(f"Error sending goal: {e}")
            self.connected = False
            return False

    def close(self):
        if self.sock:
            self.sock.close()
            self.connected = False


