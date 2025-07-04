import pyrealsense2 as rs
import numpy as np
import cv2
import time
import logging
import scipy.io as scio
import open3d as o3d
import sys
import os

sys.path.append('/workspace/ros2/src/graspnet-baseline/utils')
from data_utils import CameraInfo, create_point_cloud_from_depth_image

from typing import Tuple

class CameraRealsense:
    def __init__(self, serial_number: str=None):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._pipeline_profile = None
        self._align = None
        self._intrinsics = None
        self._depth_scale = None
        self._connected = False
        self._serial_number = serial_number

    def connect(
        self,
        rgb_resolution: Tuple[int, int]=(1280, 720),
        depth_resolution: Tuple[int, int]=(1280, 720),
        fps: int=15
    ):
        if depth_resolution[0] != rgb_resolution[0] or depth_resolution[1] != rgb_resolution[1]:
            depth_resolution = rgb_resolution
            self._logger.warning(f"Depth resolution set to RGB resolution: {depth_resolution}")

        self._logger.info("Connecting to RealSense camera...")
        if self._serial_number:
            self._config.enable_device(self._serial_number)

        pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
        self._pipeline_profile = self._config.resolve(pipeline_wrapper)
        device = self._pipeline_profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        self._logger.info(f"Device connected: {device_name} (Serial: {self._serial_number})")

        # Enable color and depth streams
        self._config.enable_stream(rs.stream.color, rgb_resolution[0], rgb_resolution[1], rs.format.bgr8, fps)
        self._logger.info(f"device_name: {device_name}")

        # For D435i, D405, etc.
        if device_name in ["Intel RealSense D435I", "Intel RealSense D405"]:
            self._config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, fps)
            self._profile = self._pipeline.start(self._config)

            # Retrieve the depth sensor/scale
            self._depth_sensor = self._profile.get_device().first_depth_sensor()
            self._depth_scale = self._depth_sensor.get_depth_scale()
            self._logger.info(f"Depth scale: {self._depth_scale}")

            # OPTIONAL: Example manual exposure if camera is D405
            if device_name == "Intel RealSense D435I":
                self._depth_sensor.set_option(rs.option.exposure, 750)
                self._depth_sensor.set_option(rs.option.gain, 16)

            for sensor in device.query_sensors():
                    sensor_name = sensor.get_info(rs.camera_info.name)
                    if sensor_name == "RGB Camera":
                        sensor.set_option(rs.option.enable_auto_exposure, 0)
                        sensor.set_option(rs.option.enable_auto_white_balance, 0)
                        sensor.set_option(rs.option.exposure, 500)  
                        sensor.set_option(rs.option.gain, 16)
                        sensor.set_option(rs.option.white_balance, 5500)
                        break
            for sensor in self._profile.get_device().query_sensors():
                if sensor.get_info(rs.camera_info.name) == "Stereo Module":
                    sensor.set_option(rs.option.emitter_enabled, 1)
                    sensor.set_option(rs.option.laser_power, 360)   # max for D435i
                    break        

            # Align depth to color
            self._align_to = rs.stream.color
            self._align = rs.align(self._align_to)
            for _ in range(45):
                self._pipeline.wait_for_frames()
            # Read frames once to get color intrinsics for aligned depth
            frames = self._pipeline.wait_for_frames()
            aligned_frames = self._align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Get intrinsics from the COLOR frame, matching the aligned depth
            color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            self._intrinsics = color_intrinsics  # Use color intrinsics

            self._connected = True
        else:
            raise Exception("Unsupported camera model")

    def camera_k(self) -> np.ndarray:
        """Return 3×3 intrinsics matrix (for the color camera, aligned depth)."""
        if not self._connected:
            raise Exception("Camera not connected. Call connect() first.")
        return np.array([
            [self._intrinsics.fx, 0,                  self._intrinsics.ppx],
            [0,                  self._intrinsics.fy, self._intrinsics.ppy],
            [0,                  0,                  1]
        ], dtype=np.float32)

    def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            color_image: H×W×3 (BGR, uint8)
            depth_image: H×W float, in meters
        """
        if not self._connected:
            raise Exception("Camera not connected. Call connect() first.")

        frames = self._pipeline.wait_for_frames()
        aligned_frames = self._align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            raise RuntimeError("Could not retrieve aligned frames")
        

        # Depth in meters
        depth_image = np.asanyarray(aligned_depth_frame.get_data()) * self._depth_scale
        # Color in BGR
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

    def __del__(self):
        if self._connected:
            self._pipeline.stop()

####################################
# Main: Capture, Save, Then Re-Load + Show Point Cloud
####################################
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    camera = CameraRealsense(serial_number='141222071355')
    camera.connect()

    color_img, depth_img_m = camera.get_rgbd()
    if color_img is None or depth_img_m is None:
        logging.warning("Failed to capture images.")
        exit()

    # 1) Save color image (BGR) as PNG
    timestamp = int(time.time())
    color_filename = f"rgb_{timestamp}.png"
    cv2.imwrite(color_filename, color_img)
    logging.info(f"Saved color image: {color_filename}")

    # 2) Save depth as 16-bit PNG in millimeters
    depth_mm = (depth_img_m * 1000).astype(np.uint16)
    depth_filename = f"depth_{timestamp}.png"
    cv2.imwrite(depth_filename, depth_mm)
    logging.info(f"Saved depth image: {depth_filename}")

    # 3) Write meta.mat with intrinsics + factor_depth
    intrinsics = camera.camera_k()  # shape (3,3)
    factor_depth_val = 1000       # 1000 => convert mm <-> m
    meta_filename = f"meta_{timestamp}.mat"
    scio.savemat(meta_filename, {
        'intrinsic_matrix': intrinsics,
        'factor_depth': np.array([[factor_depth_val]], dtype=np.float32)
    })
    logging.info(f"Saved {meta_filename} with intrinsics and factor_depth={factor_depth_val}")

    del camera  # stop the pipeline
