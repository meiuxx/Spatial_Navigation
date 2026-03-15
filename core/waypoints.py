import json
import socket
import time

def load_waypoints(json_path="waypoints.json"):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        waypoints = data.get("waypoints", [])
        return [(wp["x"], wp["y"]) for wp in waypoints]
    except Exception as e:
        print(f"Error loading waypoints: {e}")
        return []

def send_command(sock, command, expect_done=False):
    """Send a command and return the response(s). If expect_done, read until 'DONE'."""
    try:
        sock.sendall((command + "\n").encode())
        if not expect_done:
            # Simple case: just read one line (OK/ERROR)
            response = sock.recv(1024).decode().strip()
            print(f"Sent: {command} | Response: {response}")
            return response
        else:
            # For MOVE, we expect OK immediately, then later DONE
            # First read the immediate OK
            response = sock.recv(1024).decode().strip()
            print(f"Sent: {command} | Immediate: {response}")
            if response != "OK":
                return response  # error

            # Now wait for DONE (could be multiple lines, but we expect exactly DONE)
            while True:
                chunk = sock.recv(1024).decode().strip()
                if not chunk:
                    break
                print(f"Received: {chunk}")
                if chunk == "DONE":
                    return "DONE"
                # In case of extra data, we might need to handle but simple case works.
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    unity_host = "127.0.0.1"
    unity_port = 5002
    json_file = "waypoints.json"

    waypoints = load_waypoints(json_file)
    if not waypoints:
        print("No waypoints to send.")
        return

    print(f"Loaded {len(waypoints)} waypoints:")
    for i, (x, z) in enumerate(waypoints):
        print(f"  {i+1}: ({x:.2f}, {z:.2f})")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((unity_host, unity_port))
        print(f"Connected to Unity at {unity_host}:{unity_port}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    for idx, (x, z) in enumerate(waypoints):
        print(f"\nSending waypoint {idx+1}/{len(waypoints)}: ({x:.2f}, {z:.2f})")
        cmd = f"MOVE {x} {z}"
        result = send_command(sock, cmd, expect_done=True)
        if result != "DONE":
            print(f"Error or unexpected response: {result}. Stopping.")
            break

    print("All waypoints processed. Closing connection.")
    sock.close()

if __name__ == "__main__":
    main()