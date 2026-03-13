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
            
        map_np = np.frombuffer(self.mapbytes, dtype=np.uint8) # take it from buffer instead of from variable
        map_np = map_np.reshape((self.map_size_pixels, self.map_size_pixels))
        map_color = cv2.cvtColor(map_np, cv2.COLOR_GRAY2BGR) # convert to bgr for functions like circle and line
                
        scale = self.map_size_pixels / self.map_size_meters

        robot_x = int(self.agent_pose[0] / 1000.0 * scale) # DONT CENTER SHIFT THEM
        robot_y = int(self.agent_pose[1] / 1000.0 * scale)
        
        cv2.circle(map_color, (robot_x, robot_y), 20, (0, 0, 255), -1) # patched!!
        
        return map_color


    def save_map(self, filename='slam_map'):
        map_image = self.get_map_image()
        if map_image is not None:
            cv2.imwrite(f'{filename}.png', map_image)
            print(f"Map saved to {filename}.png")
            
            # heed advice to save raw map
            with open(f'{filename}.pgm', 'wb') as f:
                f.write(b'P5\n')
                f.write(f'{self.map_size_pixels} {self.map_size_pixels}\n'.encode())
                f.write(b'255\n')
                f.write(self.mapbytes)
            print(f"Raw map saved to {filename}.pgm")

    def get_map_array(self):
        # for frontier detection
        map_np = np.frombuffer(self.mapbytes, dtype = np.uint8).copy()
        map_np = map_np.reshape(self.map_size_pixels, self.map_size_pixels)
        return map_np