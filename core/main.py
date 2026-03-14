import socket
import navigation.occupancy_grid as og
import cv2
import time
import json
import numpy as np
import navigation.frontier_detection
import navigation.utils
import navigation.commander


def run_slam(host='127.0.0.1', port=5001):
    # Create server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"listening on {host}:{port}")

    # Initialize SLAM processor
    slam_processor = og.SLAMProcess(map_size_pixels=5000, map_size_meters=100)

    conn = None
    try:
        # Accept connection
        conn, addr = server_socket.accept()
        cmd_sender = navigation.commander.CommandSender(unity_host='127.0.0.1', command_port=5002)

        # After creating cmd_sender
        conn.settimeout(2.0)
        print(f"Connected to Unity at {addr}")

        buffer = ""
        last_visualization_time = time.time()
        visualization_interval = 0.5  # Update display every 0.5 seconds

        # Create OpenCV window
        cv2.namedWindow('SLAM Map', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SLAM Map', 600, 600)

        print("\n" + "=" * 60)
        print("SLAM ACTIVE")
        print("=" * 60 + "\n")

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
                        print(f"Invalid JSON received")
                    except Exception as e:
                        print(f"Processing error: {e}")

                # Update visualization at regular intervals
                current_time = time.time()
                if current_time - last_visualization_time >= visualization_interval:
                    last_visualization_time = current_time
                    
                    # 1. Update Visualization (Keep this frequent for the UI)
                    map_image = slam_processor.get_map_image()
                    if map_image is not None:
                        cv2.imshow('SLAM Map', map_image)
                        cv2.waitKey(1)

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
        print("\nShutting down...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup
        if conn:
            conn.close()
        cmd_sender.close()
        server_socket.close()

        # Save final map
        slam_processor.save_map('final_slam_map')
        print("final map saved")

        # Close OpenCV window
        cv2.destroyAllWindows()
        print("server shut down.")


if __name__ == "__main__":
    print("=" * 60)
    print("Unity ↔ Python LiDAR SLAM Bridge with Visualization")
    print("=" * 60)
    run_slam()