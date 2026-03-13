# slam/utils.py
import numpy as np

def quantize_map(map_array, occupied_thresh=85, free_thresh=170):
    quantized = np.zeros_like(map_array, dtype=np.int8)
    quantized[map_array < occupied_thresh] = 1 #this is occupied
    quantized[(map_array >= occupied_thresh) & (map_array < free_thresh)] = -1
    quantized[map_array >= free_thresh] = 0 #this is free
    return quantized

def pixels_to_world(pixel_x, pixel_y, map_size_pixels, map_size_meters):
    scale = map_size_meters / map_size_pixels
    world_x = (pixel_x - map_size_pixels / 2) * scale
    world_y = (map_size_pixels / 2 - pixel_y) * scale
    return world_x, world_y

def world_to_pixels(world_x, world_y, map_size_pixels, map_size_meters):
    scale = map_size_pixels / map_size_meters
    pixel_x = int(world_x * scale + map_size_pixels / 2)
    pixel_y = int(map_size_pixels / 2 - world_y * scale)
    return pixel_x, pixel_y
