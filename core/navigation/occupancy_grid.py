import json # to parse messages from Unity, they're in JSON format
import socket #creates a tcp server that listens to unity's connection
import numpy as np # for array operations
import time # for viz update rate
import cv2 # for real time viz

from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser

class SLAMProcess:

    def __init__(self, map_size_pixels=800, map_size_meters=35):
        
        self.lidar_model = Laser(
            scan_size=360,
            scan_rate_hz=20,
            detection_angle_degrees=360,
            distance_no_detection_mm=30000
        )

        self.slam = RMHC_SLAM(self.lidar_model, map_size_pixels, map_size_meters)
        self.map_size_pixels = map_size_pixels
        self.map_size_meters = map_size_meters
        self.mapbytes = bytearray(map_size_pixels*map_size_pixels) # so we can later divide it
        self.agent_pose = [0.0, 0.0, 0.0]
        self.scan_count = 0


    # parse the input from unity. Unity outputs json strings in the format:
    # {”odom”: { “dx”: 0.05, “dy”: 0.0, dtheta: 1.2 }, “ranges”: [3.5, 2.1, ….]}
    def parse_message(self, message):
        # load message into a python object
        # if it already is a python object then use as is
        data = json.loads(message) if isinstance(message, str) else message

        if 'odom' in data:
            self.latest_odom = (
                data['odom']['dx'] * 1000.0,   # Unity forward (Z) -> SLAM X
                data['odom']['dy'] * 1000.0,   # Unity right (X)   -> SLAM Y
                data['odom']['dtheta']          # heading change (degrees)
            )
        else:
            self.latest_odom = (0.0, 0.0, 0.0)

        if 'ranges_mm' in data:
            return data['ranges_mm']
        
        #if send meters
        if 'ranges' in data:
            scan_mm = []
            for dist in data['ranges']:
                if dist == float('inf') or dist >= 30.0:
                    scan_mm.append(self.lidar_model.distance_no_detection_mm)
                else:
                    scan_mm.append(int(dist * 1000))
            return scan_mm


    
    def update(self, scan_data):
        # to protect against malformed data:
        if not scan_data or len(scan_data) != self.lidar_model.scan_size:
            return None
        
        self.scan_count+=1
        self.slam.update(scan_data, pose_change=self.latest_odom)

        x_mm, y_mm, theta_degrees = self.slam.getpos()
        self.agent_pose = [x_mm, y_mm, theta_degrees]
        
        self.slam.getmap(self.mapbytes)
        
        return self.agent_pose


    def get_map_image(self):
        if not self.mapbytes:
            return None
            
        map_np = np.frombuffer(self.mapbytes, dtype=np.uint8)
        map_np = map_np.reshape((self.map_size_pixels, self.map_size_pixels))
        map_color = cv2.cvtColor(map_np, cv2.COLOR_GRAY2BGR)
        
        scale = self.map_size_pixels / self.map_size_meters
        robot_x = int(self.agent_pose[0] / 1000.0 * scale + self.map_size_pixels / 2)
        robot_y = int(self.agent_pose[1] / 1000.0 * scale + self.map_size_pixels / 2)
        
        cv2.circle(map_color, (robot_x, robot_y), 5, (0, 0, 255), -1)
        
        orientation_length = 20
        angle_rad = np.radians(self.agent_pose[2])
        end_x = int(robot_x + orientation_length * np.cos(angle_rad))
        end_y = int(robot_y - orientation_length * np.sin(angle_rad))
        
        cv2.line(map_color, (robot_x, robot_y), (end_x, end_y), (0, 255, 0), 2)
        
        return map_color


    def save_map(self, filename='slam_map'):
        map_image = self.get_map_image()
        if map_image is not None:
            cv2.imwrite(f'{filename}.png', map_image)
            print(f"Map saved to {filename}.png")
            
            with open(f'{filename}.pgm', 'wb') as f:
                f.write(b'P5\n')
                f.write(f'{self.map_size_pixels} {self.map_size_pixels}\n'.encode())
                f.write(b'255\n')
                f.write(self.mapbytes)
            print(f"Raw map saved to {filename}.pgm")
    

def run_slam(host='127.0.0.1', port=5001):
    """Socket server with real-time map visualization"""
    
    # Create server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    # Initialize SLAM processor
    slam_processor = SLAMProcess(map_size_pixels=5000, map_size_meters=100)
    
    conn = None
    try:
        # Accept connection
        conn, addr = server_socket.accept()
        conn.settimeout(2.0)
        print(f"Connected to Unity at {addr}")
        
        buffer = ""
        last_visualization_time = time.time()
        visualization_interval = 0.5  # Update display every 0.5 seconds
        
        # Create OpenCV window
        cv2.namedWindow('SLAM Map', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SLAM Map', 600, 600)
        
        print("\n" + "="*60)
        print("SLAM ACTIVE - Move your robot in Unity to build a map!")
        print("="*60 + "\n")
        
        while True:
            try:
                # Check for OpenCV window close
                if cv2.getWindowProperty('SLAM Map', cv2.WND_PROP_VISIBLE) < 1:
                    print("Map window closed by user")
                    break
                
                # Receive data
                data = conn.recv(4096)
                if not data:
                    print("Unity disconnected.")
                    break
                
                # Decode and add to buffer
                buffer += data.decode('utf-8', errors='ignore')
                
                # Process complete messages
                messages = []
                while '\n' in buffer:
                    message, buffer = buffer.split('\n', 1)
                    if message.strip():
                        messages.append(message.strip())
                
                # Process all received messages
                for message in messages:
                    try:
                        # Parse and process the LiDAR scan
                        scan_mm = slam_processor.parse_message(message)
                        
                        if scan_mm:
                            # Update SLAM
                            pose = slam_processor.update(scan_mm)
                            
                            if pose:
                                # Print pose occasionally
                                if slam_processor.scan_count % 20 == 0:
                                    print(f"Scan #{slam_processor.scan_count}: "
                                          f"Pos({pose[0]/1000:.2f}m, {pose[1]/1000:.2f}m), "
                                          f"Heading: {pose[2]:.1f}°")
                    
                    except json.JSONDecodeError:
                        print(f"⚠️ Invalid JSON received")
                    except Exception as e:
                        print(f"⚠️ Processing error: {e}")
                
                # Update visualization at regular intervals
                current_time = time.time()
                if current_time - last_visualization_time >= visualization_interval:
                    map_image = slam_processor.get_map_image()
                    if map_image is not None:
                        # Add text overlay
                        cv2.putText(map_image, f"Scans: {slam_processor.scan_count}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(map_image, "Black=Obstacle, White=Free, Gray=Unknown", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Show the map
                        cv2.imshow('SLAM Map', map_image)
                        cv2.waitKey(1)  # Brief wait to update display
                    
                    last_visualization_time = current_time
                
            except socket.timeout:
                # Timeout is ok, just check for window and continue
                continue
            except ConnectionResetError:
                print("Unity connection was reset.")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                break
                
    except KeyboardInterrupt:
        print("\n👋 Shutting down by user request...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup
        if conn:
            conn.close()
        server_socket.close()
        
        # Save final map
        try:
            slam_processor.save_map('final_slam_map')
            print("💾 Final map saved!")
        except Exception as e:
            print(f"Could not save map: {e}")
        
        # Close OpenCV window
        cv2.destroyAllWindows()
        print("Server shut down.")

if __name__ == "__main__":
    print("=" * 60)
    print("Unity ↔ Python LiDAR SLAM Bridge with Visualization")
    print("=" * 60)
    run_slam()