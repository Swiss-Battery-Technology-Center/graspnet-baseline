import pyrealsense2 as rs
import numpy as np
import os
import logging
import time
import traceback
import cv2

class CameraRealsense:
    def __init__(self, serial_number=None, logger=None):
        self._logger = logger or logging.getLogger("CameraRealsense")
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._serial_number = serial_number
        self._depth_scale = None
        self._align = None
        self._connected = False
        self._intrinsics = None
        self._rgb_resolution = (1280, 720)
        self._fps = 15

    def connect(self):
        if self._connected:
            self._logger.warning("Camera already connected.")
            return True

        try:
            if self._serial_number:
                self._config.enable_device(self._serial_number)
            self._config.enable_stream(rs.stream.depth, *self._rgb_resolution, rs.format.z16, self._fps)
            self._config.enable_stream(rs.stream.color, *self._rgb_resolution, rs.format.bgr8, self._fps)

            profile = self._pipeline.start(self._config)
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()
            self._logger.info(f"Depth scale: {self._depth_scale}")

            self._align = rs.align(rs.stream.color)
            time.sleep(1.5)

            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self._intrinsics = color_profile.get_intrinsics()

            self._connected = True
            self._logger.info("Camera connected.")
            return True
        except Exception as e:
            self._logger.error(f"Failed to connect to camera: {e}")
            traceback.print_exc()
            return False

    def get_rgbd_k(self):
        if not self._connected:
            self._logger.error("Camera not connected.")
            return None, None, None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self._align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                self._logger.error("Failed to get frames.")
                return None, None, None
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self._depth_scale
            color = np.asanyarray(color_frame.get_data())

            K = np.array([
                [self._intrinsics.fx, 0, self._intrinsics.ppx],
                [0, self._intrinsics.fy, self._intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float32)

            return color, depth, K
        except Exception as e:
            self._logger.error(f"Failed to retrieve frames: {e}")
            traceback.print_exc()
            return None, None, None

    def disconnect(self):
        if self._connected:
            try:
                self._pipeline.stop()
                self._connected = False
                self._logger.info("Camera disconnected.")
            except Exception as e:
                self._logger.error(f"Error while disconnecting: {e}")

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("CaptureRGBDToNPZ")

    cam = CameraRealsense(logger=logger)
    if not cam.connect():
        return

    color, depth, K = cam.get_rgbd_k()
    if color is not None and depth is not None and K is not None:
        os.makedirs("recorded_frames", exist_ok=True)
        
        # Save .npz
        np.savez("recorded_frames/rgbd_frame.npz", rgb=color, depth=depth, K=K)
        logger.info("Saved RGB, depth, and intrinsics K to recorded_frames/rgbd_frame.npz.")

        # Save RGB as PNG
        rgb_png_path = "recorded_frames/rgb_image.png"
        cv2.imwrite(rgb_png_path, color)
        logger.info(f"Saved RGB image as PNG to {rgb_png_path}")
    else:
        logger.error("Failed to capture and save frame.")

    cam.disconnect()

if __name__ == "__main__":
    main()