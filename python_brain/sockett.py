import socket
import struct
import json
from PIL import Image
import io
import os
import traceback

HOST = "127.0.0.1"
PORT = 5001
SAVE_DIR = "observations"
os.makedirs(SAVE_DIR, exist_ok=True)

def recv_exact(sock, n):
    """Receive exactly n bytes from socket"""
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Socket connection lost")
        data += packet
    return data

def save_image(img_bytes, step):
    """Save image bytes to file"""
    try:
        image = Image.open(io.BytesIO(img_bytes))
        filename = os.path.join(SAVE_DIR, f"frame_{step:05d}.png")
        image.save(filename)
        return filename
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def debug_bytes(data, label, max_len=100):
    """Debug helper to print byte data"""
    print(f"\n{label} ({len(data)} bytes):")
    if len(data) <= max_len:
        print(f"Hex: {data.hex()}")
        print(f"ASCII: {data[:max_len]}")
    else:
        print(f"First {max_len} bytes hex: {data[:max_len].hex()}")
        print(f"First {max_len} bytes ASCII: {data[:max_len]}")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print(f"[Python] TCP server listening on {HOST}:{PORT}")

conn, addr = server.accept()
print(f"[Python] Connected by {addr}")

step_counter = 0  # Track steps independently

try:
    while True:
        try:
            print(f"\n{'='*50}")
            print(f"Waiting for data (Python step counter: {step_counter})...")
            
            # Step 1: Read lengths (4 bytes each for JSON and image)
            lengths_bytes = recv_exact(conn, 8)
            debug_bytes(lengths_bytes, "Lengths header")
            
            # Unpack lengths - ensure correct byte order
            json_len, img_len = struct.unpack("<ii", lengths_bytes)  # Little-endian
            print(f"JSON length: {json_len}, Image length: {img_len}")
            
            if json_len <= 0 or img_len <= 0:
                print(f"WARNING: Invalid lengths - JSON: {json_len}, Image: {img_len}")
                continue
            
            # Step 2: Read JSON data
            json_bytes = recv_exact(conn, json_len)
            debug_bytes(json_bytes[:min(200, len(json_bytes))], "JSON data (first 200 bytes)")
            
            # Try to decode JSON
            json_str = json_bytes.decode('utf-8', errors='ignore')
            print(f"JSON string: {json_str[:200]}...")
            
            # Parse JSON
            try:
                obs = json.loads(json_str)
                print(f"Parsed JSON keys: {list(obs.keys())}")
                
                # Extract data with fallbacks
                step = obs.get("step", obs.get("Step", obs.get("frameId", step_counter)))
                rays = obs.get("rays", obs.get("Rays", obs.get("detectedObjects", [])))
                timestamp = obs.get("timestamp", obs.get("Timestamp", ""))
                
                print(f"\n=== Step {step} ===")
                print(f"Timestamp: {timestamp}")
                print(f"Rays count: {len(rays)}")
                
                if rays:
                    print("First few ray detections:")
                    for i, ray in enumerate(rays[:3]):
                        print(f"  {i}: {ray}")
                    if len(rays) > 3:
                        print(f"  ... and {len(rays) - 3} more")
                else:
                    print("No ray detections")
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw data: {json_str[:100]}")
                step = step_counter
                rays = []
            
            # Step 3: Read image data
            print(f"Reading {img_len} image bytes...")
            img_bytes = recv_exact(conn, img_len)
            
            # Debug image header (PNG should start with PNG signature)
            print(f"Image bytes received: {len(img_bytes)}")
            if len(img_bytes) >= 8:
                print(f"Image header: {img_bytes[:8].hex()}")
                if img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                    print("✓ Valid PNG header")
                else:
                    print("⚠ Unexpected image header")
            
            # Step 4: Save image
            if img_bytes and len(img_bytes) > 100:  # Minimum reasonable size
                frame_file = save_image(img_bytes, step)
                if frame_file:
                    print(f"Saved frame: {frame_file}")
                else:
                    print("Failed to save image")
            else:
                print(f"Invalid image data (size: {len(img_bytes)})")
            
            step_counter += 1
            
        except struct.error as e:
            print(f"Struct unpack error: {e}")
            print("This usually means the data format doesn't match what we expect")
            break
        except ConnectionError as e:
            print(f"Connection error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            break

except KeyboardInterrupt:
    print("\n[Python] Shutting down server...")
except Exception as e:
    print(f"Fatal error: {e}")
    traceback.print_exc()
finally:
    conn.close()
    server.close()
    print("[Python] Server closed")