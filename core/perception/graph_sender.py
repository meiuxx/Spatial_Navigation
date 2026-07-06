import socket
import json
import time
import threading
import queue

class GraphSender:
    def __init__(self, host='127.0.0.1', port=5006, send_interval=2.0):
        self.host = host
        self.port = port
        self.send_interval = send_interval  # minimum seconds between sends

        self.sock = None
        self.connected = False
        self._lock = threading.Lock()

        # Single-slot queue — only the latest graph matters
        # If a send is pending and a newer graph arrives, discard the old one
        self._pending = None
        self._pending_lock = threading.Lock()

        self._last_sent = 0

        # Background thread handles all socket work — never blocks main thread
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────────

    def send_graph(self, graph):
        """Non-blocking. Drops the serialised graph in the pending slot;
        the worker thread sends it once the interval has elapsed. Any
        unsent previous graph is replaced -- only the latest state matters."""
        now = time.time()
        if now - self._last_sent < self.send_interval:
            return

        try:
            data = self._serialise(graph)
        except Exception as e:
            print(f"[GraphSender] Serialisation error: {e}")
            return

        with self._pending_lock:
            self._pending = data

    def close(self):
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass

    # ── Background worker ─────────────────────────────────────────────────

    def _worker(self):
        """Connects to Unity and loops sending pending graphs, reconnecting
        on failure. Runs on a daemon thread, never blocks the main thread."""
        while True:
            if not self.connected:
                self._try_connect()
                time.sleep(2.0)
                continue

            # Check for pending graph
            with self._pending_lock:
                payload = self._pending
                self._pending = None

            if payload is not None:
                success = self._send_raw(payload)
                if success:
                    self._last_sent = time.time()
                    print(f"[GraphSender] Graph sent ({len(payload)} bytes)")
                else:
                    # Put it back so it retries after reconnect
                    with self._pending_lock:
                        self._pending = payload
                    self.connected = False
                    print("[GraphSender] Send failed — will reconnect")
            else:
                # Nothing to send — sleep briefly to avoid busy-looping
                time.sleep(0.1)

    def _try_connect(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((self.host, self.port))
            sock.settimeout(5.0)  # generous timeout for sends
            with self._lock:
                self.sock = sock
            self.connected = True
            print(f"[GraphSender] Connected to Unity on port {self.port}")
        except Exception:
            print(f"[GraphSender] Unity not ready on port {self.port}, retrying...")
            self.connected = False

    def _send_raw(self, payload):
        try:
            with self._lock:
                self.sock.sendall(payload.encode('utf-8'))
            return True
        except Exception as e:
            print(f"[GraphSender] Socket error: {e}")
            return False

    # ── Serialisation ─────────────────────────────────────────────────────

    def _serialise(self, graph):
        data = {"nodes": [], "edges": []}
        for nid, attrs in graph.nodes(data=True):
            pos = attrs.get('position', [0, 0, 0])
            data["nodes"].append({
                "id": nid,
                "pos": [round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)],
                "label": attrs.get('clip_label', '?'),
                "activation": round(attrs.get('activation', 1.0), 3)
            })
        for u, v in graph.edges():
            data["edges"].append({"from": u, "to": v})
        return json.dumps(data) + "\n"