import socket
import perception.process as perception
from perception.graph_sender import GraphSender

sender = GraphSender(host='127.0.0.1', port=5006)

def handle_client(conn, addr):
    print(f"Client connected from {addr}")
    buffer = ""
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                print("Client closed connection")
                break
            buffer += data.decode('utf-8')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line.strip():
                    # Inside the message processing loop:
                    perception.process_message(line.strip())
                    sender.send_graph(perception.semantic_mapper.graph)
    
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        conn.close()
        print("Client disconnected")

def start_server(host='127.0.0.1', port=5004):
    """Start TCP server, accept one client at a time."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"Server listening on {host}:{port}")

    while True:
        conn, addr = server.accept()
        handle_client(conn, addr)

if __name__ == "__main__":
    print("Starting main.py – connection only")
    start_server()