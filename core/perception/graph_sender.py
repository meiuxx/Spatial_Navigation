import socket
import json
import time
import threading

class GraphSender:
    def __init__(self, host='127.0.0.1', port=5006, auto_send_interval=2.0):
        self.host = host
        self.port = port
        self.auto_send_interval = auto_send_interval
        self.sock = None
        self.connected = False
        self.lock = threading.Lock()
        self._connect()
        if auto_send_interval > 0:
            self._start_auto_send()

    def _connect(self):
        while not self.connected:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # Set a timeout so it doesn't hang forever on one attempt
                self.sock.settimeout(2.0) 
                self.sock.connect((self.host, self.port))
                self.connected = True
                print(f"[GraphSender] Connected to Unity on port {self.port}")
            except Exception:
                print(f"[GraphSender] Waiting for Unity on port {self.port}...")
                if self.sock:
                    self.sock.close()
                time.sleep(2.0) # Wait 2 seconds before trying again

    def send_graph(self, graph):
        """Send the networkx graph as JSON."""
        if not self.connected:
            return False
        # Convert graph to JSON format expected by Unity
        data = {"nodes": [], "edges": []}
        id_to_index = {}
        for nid, attrs in graph.nodes(data=True):
            pos = attrs.get('position', [0,0,0])
            label = attrs.get('clip_label', '?')
            data["nodes"].append({
                "id": nid,
                "pos": [pos[0], pos[1], pos[2]],
                "label": label
            })
            id_to_index[nid] = len(data["nodes"]) - 1   # store index for later
        # Edges (only temporal edges for simplicity, or all)
        for u, v in graph.edges():
            data["edges"].append({"from": u, "to": v})
        msg = json.dumps(data) + "\n"
        print(f"Sending graph: {msg[:200]}...")   # first 200 chars
        try:
            with self.lock:
                self.sock.sendall(msg.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False

    def _start_auto_send(self):
        def sender_loop():
            while self.connected:
                time.sleep(self.auto_send_interval)
                # You need access to the graph. We'll assume you pass it via a callback.
                # For simplicity, we can have a global variable or a method to set the graph.
                # Let's leave it as a placeholder; you'll call send_graph explicitly.
        threading.Thread(target=sender_loop, daemon=True).start()

    def close(self):
        if self.sock:
            self.sock.close()
            self.connected = False