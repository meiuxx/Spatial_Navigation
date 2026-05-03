import socket
import perception.process as perception
from perception.graph_sender import GraphSender

# GraphSender starts its background worker immediately but
# doesn't block — it quietly retries until Unity is ready
sender = GraphSender(host='127.0.0.1', port=5006, send_interval=2.0)

def handle_client(conn, addr):
    print(f"[Server] Client connected from {addr}")
    buffer = ""
    frame_count = 0
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                print("[Server] Client disconnected cleanly")
                break
            buffer += data.decode('utf-8')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if line:
                    frame_count += 1
                    perception.process_message(line)
                    # Non-blocking — never stalls the receive loop
                    sender.send_graph(perception.semantic_mapper.graph)

    except Exception as e:
        print(f"[Server] Connection error: {e}")
    finally:
        conn.close()
        print(f"[Server] Disconnected after {frame_count} frames")

def start_server(host='127.0.0.1', port=5004):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[Server] Listening on {host}:{port} — waiting for Unity...")
    while True:
        conn, addr = server.accept()
        handle_client(conn, addr)

if __name__ == "__main__":
    start_server()