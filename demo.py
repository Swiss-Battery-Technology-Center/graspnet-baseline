#!/usr/bin/env python3

import os
import sys
import numpy as np
import open3d as o3d
import argparse
from scipy.spatial.transform import Rotation as R # Used for conversion AND multiplication
import time
import logging
import traceback

# --- Camera Imports ---
import pyrealsense2 as rs
import cv2
from typing import Tuple, Optional

# --- ROS 2 Imports ---
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import TransformStamped, Quaternion
import tf2_ros
from std_srvs.srv import Trigger
import rclpy.logging # For LoggingSeverity enum if needed elsewhere

# --- PyTorch Import ---
import torch

# --- GraspNet Imports ---
# ... (imports as before) ...
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
try:
    from graspnet import GraspNet, pred_decode
    grasp_api_path_util = os.path.join(ROOT_DIR, 'utils', 'graspnetAPI')
    grasp_api_path_root = os.path.join(ROOT_DIR, 'graspnetAPI')
    if os.path.exists(grasp_api_path_util):
        sys.path.append(os.path.join(ROOT_DIR, 'utils'))
        from graspnetAPI import GraspGroup, Grasp
    elif os.path.exists(grasp_api_path_root):
        sys.path.append(grasp_api_path_root)
        from graspnetAPI import GraspGroup, Grasp
    else: from graspnetAPI import GraspGroup, Grasp
    from collision_detector import ModelFreeCollisionDetector
    from data_utils import CameraInfo, create_point_cloud_from_depth_image
except ImportError as e: print(f"Import Error: {e}\nSys Path: {sys.path}"); sys.exit(1)


# =============================================================================
# RealSense Camera Class (Using Reverted Settings)
# =============================================================================
class CameraRealsense:
    # --- Using the reverted version from the previous response ---
    def __init__(self, serial_number: str=None, logger=None):
        self._logger = logger if logger else logging.getLogger(self.__class__.__name__)
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._pipeline_profile = None
        self._align = None
        self._intrinsics = None
        self._depth_scale = None
        self._connected = False
        self._serial_number = serial_number
        self._rgb_resolution = (1280, 720)
        self._depth_resolution = (1280, 720)
        self._fps = 15

    def connect(self, rgb_resolution: Tuple[int, int]=(1280, 720), depth_resolution: Tuple[int, int]=(1280, 720), fps: int=15):
        if self._connected:
            self._logger.warning("Camera already connected.")
            return True
        self._rgb_resolution=rgb_resolution
        self._depth_resolution=depth_resolution
        self._fps=fps
        if self._depth_resolution != self._rgb_resolution:
            self._logger.warning(f"Depth resolution {self._depth_resolution} differs from RGB {self._rgb_resolution}. Forcing depth to match RGB.")
            self._depth_resolution = self._rgb_resolution
        self._logger.info(f"Attempting to connect RealSense camera...")
        self._logger.info(f"  Requested RGB resolution: {self._rgb_resolution}")
        self._logger.info(f"  Requested Depth resolution: {self._depth_resolution}")
        self._logger.info(f"  Requested FPS: {self._fps}")
        if self._serial_number:
            self._logger.info(f"  Targeting serial number: {self._serial_number}")
            self._config.enable_device(self._serial_number)
        else: self._logger.info("  No specific serial number targeted.")
        try:
            self._config.enable_stream(rs.stream.depth, *self._depth_resolution, rs.format.z16, self._fps)
            self._config.enable_stream(rs.stream.color, *self._rgb_resolution, rs.format.bgr8, self._fps)
            pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
            self._pipeline_profile = self._config.resolve(pipeline_wrapper)
            device = self._pipeline_profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            actual_serial = device.get_info(rs.camera_info.serial_number)
            self._logger.info(f"Successfully resolved configuration for device: {device_name} (Serial: {actual_serial})")
            depth_sensor = device.first_depth_sensor()
            if not depth_sensor: self._logger.error("Could not get depth sensor from device."); return False
            self._depth_scale = depth_sensor.get_depth_scale()
            self._logger.info(f"Depth scale: {self._depth_scale}")
            self._logger.info(f"Checking if specific settings are needed for: {device_name}")
            if "D435" in device_name or "D435I" in device_name:
                self._apply_d435_settings(device, depth_sensor) # Uses the reverted fixed settings method below
            else: self._logger.info(f"No specific settings applied for {device_name}.")
            self._profile = self._pipeline.start(self._config)
            self._logger.info("Pipeline started.")
            self._align_to = rs.stream.color
            self._align = rs.align(self._align_to)
            self._logger.info(f"Alignment configured to stream: {self._align_to}")
            self._logger.info("Allowing camera stream to stabilize...")
            time.sleep(1.5)
            color_profile = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
            if not color_profile: self._logger.error("Could not get color profile after starting pipeline."); self.disconnect(); return False
            self._intrinsics = color_profile.get_intrinsics()
            self._logger.info(f"Intrinsics obtained: fx={self._intrinsics.fx}, fy={self._intrinsics.fy}, ppx={self._intrinsics.ppx}, ppy={self._intrinsics.ppy}")
            self._connected = True
            self._logger.info("RealSense camera connected and configured successfully.")
            return True
        except RuntimeError as e: self._logger.error(f"RuntimeError during camera connection: {e}"); traceback.print_exc(); self._connected = False; return False
        except Exception as e: self._logger.error(f"Unexpected error during camera connection: {e}"); traceback.print_exc(); self._connected = False; return False

    # --- REVERTED D435 FIXED SETTINGS METHOD ---
    def _apply_d435_settings(self, device, depth_sensor):
        """Applies specific fixed settings for D435/D435i cameras."""
        self._logger.info("Applying specific D435(i) options (fixed settings)...")
        try: # Stereo Module Settings
            stereo_module = next((s for s in device.query_sensors() if s.get_info(rs.camera_info.name) == 'Stereo Module'), None)
            if stereo_module:
                opt_emitter = rs.option.emitter_enabled
                if stereo_module.supports(opt_emitter) and not stereo_module.is_option_read_only(opt_emitter): stereo_module.set_option(opt_emitter, 1.0); self._logger.info("  Stereo Module: Emitter Enabled set to 1 (ON)")
                else: self._logger.warn("  Stereo Module: Cannot set Emitter Enabled.")
                opt_laser = rs.option.laser_power
                if stereo_module.supports(opt_laser) and not stereo_module.is_option_read_only(opt_laser):
                    laser_range = stereo_module.get_option_range(opt_laser); set_laser = min(360.0, laser_range.max)
                    stereo_module.set_option(opt_laser, set_laser); self._logger.info(f"  Stereo Module: Laser Power set to {set_laser} (Max possible: {laser_range.max})")
                else: self._logger.warn("  Stereo Module: Cannot set Laser Power.")
            else: self._logger.warn("Stereo Module not found on device.")
            # Depth Sensor Manual Exposure/Gain
            opt_depth_ae = rs.option.enable_auto_exposure
            if depth_sensor.supports(opt_depth_ae) and not depth_sensor.is_option_read_only(opt_depth_ae):
                 depth_sensor.set_option(opt_depth_ae, 0); self._logger.info("  Depth Sensor: Auto Exposure Disabled.")
                 opt_depth_exp = rs.option.exposure
                 if depth_sensor.supports(opt_depth_exp) and not depth_sensor.is_option_read_only(opt_depth_exp): depth_sensor.set_option(opt_depth_exp, 750.0); self._logger.info(f"  Depth Sensor: Manual Exposure set to 750.0")
                 else: self._logger.warn("  Depth Sensor: Cannot set Manual Exposure.")
                 opt_depth_gain = rs.option.gain
                 if depth_sensor.supports(opt_depth_gain) and not depth_sensor.is_option_read_only(opt_depth_gain): depth_sensor.set_option(opt_depth_gain, 16.0); self._logger.info(f"  Depth Sensor: Manual Gain set to 16.0")
                 else: self._logger.warn("  Depth Sensor: Cannot set Manual Gain.")
            else: self._logger.warn("  Depth Sensor: Cannot disable Auto Exposure to set manual values.")
        except Exception as e: self._logger.warn(f"Could not set some depth/stereo options: {e}")
        try: # RGB Sensor Manual Exposure/Gain/White Balance
            color_sensor = device.first_color_sensor()
            if color_sensor:
                opt_rgb_ae = rs.option.enable_auto_exposure
                if color_sensor.supports(opt_rgb_ae) and not color_sensor.is_option_read_only(opt_rgb_ae):
                    color_sensor.set_option(opt_rgb_ae, 0); self._logger.info("  RGB Sensor: Auto Exposure Disabled.")
                    opt_rgb_exp = rs.option.exposure
                    if color_sensor.supports(opt_rgb_exp) and not color_sensor.is_option_read_only(opt_rgb_exp): color_sensor.set_option(opt_rgb_exp, 500.0); self._logger.info(f"  RGB Sensor: Manual Exposure set to 500.0")
                    else: self._logger.warn("  RGB Sensor: Cannot set Manual Exposure.")
                    opt_rgb_gain = rs.option.gain
                    if color_sensor.supports(opt_rgb_gain) and not color_sensor.is_option_read_only(opt_rgb_gain): color_sensor.set_option(opt_rgb_gain, 16.0); self._logger.info(f"  RGB Sensor: Manual Gain set to 16.0")
                    else: self._logger.warn("  RGB Sensor: Cannot set Manual Gain.")
                else: self._logger.warn("  RGB Sensor: Cannot disable Auto Exposure to set manual values.")
                opt_rgb_awb = rs.option.enable_auto_white_balance
                if color_sensor.supports(opt_rgb_awb) and not color_sensor.is_option_read_only(opt_rgb_awb):
                     color_sensor.set_option(opt_rgb_awb, 0); self._logger.info("  RGB Sensor: Auto White Balance Disabled.")
                     opt_rgb_wb = rs.option.white_balance
                     if color_sensor.supports(opt_rgb_wb) and not color_sensor.is_option_read_only(opt_rgb_wb): color_sensor.set_option(opt_rgb_wb, 5500.0); self._logger.info(f"  RGB Sensor: Manual White Balance set to 5500.0")
                     else: self._logger.warn("  RGB Sensor: Cannot set Manual White Balance.")
                else: self._logger.warn("  RGB Sensor: Cannot disable Auto White Balance to set manual value.")
            else: self._logger.warn("RGB color sensor not found on device.")
        except Exception as e: self._logger.warn(f"Could not set some RGB options: {e}")
    # --- END REVERTED METHOD ---

    def camera_k(self) -> Optional[np.ndarray]:
        # ... (implementation as before) ...
        if not self._connected or not self._intrinsics: self._logger.error("Camera not connected or intrinsics unavailable."); return None
        k = self._intrinsics; return np.array([[k.fx, 0, k.ppx], [0, k.fy, k.ppy], [0, 0, 1]], dtype=np.float32)

    def get_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # ... (implementation as before) ...
        if not self._connected: self._logger.error("Cannot get RGBD frame, camera not connected."); return None, None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=5000);
            if not frames: self._logger.warn("No frames received within timeout period (5000 ms)."); return None, None
            aligned_frames = self._align.process(frames)
            if not aligned_frames: self._logger.warn("Frame alignment failed."); return None, None
            aligned_depth_frame = aligned_frames.get_depth_frame(); color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame: self._logger.warn("Could not retrieve aligned depth or color frame after alignment."); return None, None
            depth_raw = np.asanyarray(aligned_depth_frame.get_data()); depth_m = depth_raw.astype(np.float32) * self._depth_scale
            color_bgr = np.asanyarray(color_frame.get_data())
            if depth_m.shape[:2] != color_bgr.shape[:2]: self._logger.error(f"Aligned frame shape mismatch! Depth: {depth_m.shape}, Color: {color_bgr.shape}"); return None, None
            return color_bgr, depth_m
        except RuntimeError as e: self._logger.error(f"RuntimeError while waiting for frames: {e}"); return None, None
        except Exception as e: self._logger.error(f"Unexpected error in get_rgbd: {e}"); traceback.print_exc(); return None, None

    def disconnect(self):
        # ... (implementation as before) ...
        if self._connected:
            self._logger.info("Disconnecting RealSense camera...");
            try: self._pipeline.stop(); self._connected = False; self._intrinsics = None; self._profile = None; self._pipeline_profile = None; self._logger.info("RealSense camera disconnected.")
            except Exception as e: self._logger.error(f"Error stopping pipeline: {e}"); self._connected = False
        else: self._logger.info("Camera already disconnected.")

    @property
    def pipeline(self): return self._pipeline
    @property
    def align(self): return self._align
    @property
    def connected(self): return self._connected
    def __del__(self): self.disconnect()


# =============================================================================
# GraspNet ROS 2 Service Node
# =============================================================================
class GraspServiceNode(Node):

    # --- Define the desired FIXED rotation offsets as quaternions [x, y, z, w] ---
    # First offset (e.g., 90 deg around diagonal Y-like axis from user input)
    FIRST_ROTATION_OFFSET_QUAT = np.array([ 0.7071068, 0.0, 0.7071068, 0.0 ])
    # Second offset (-90 deg around Z axis)
    SECOND_ROTATION_OFFSET_QUAT = np.array([ 0.0, 0.0, 0.7071068, 0.7071068])
    # ---

    def __init__(self, cfgs):
        super().__init__('graspnet_service_node')
        self.cfgs = cfgs
        self.callback_group = ReentrantCallbackGroup()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.last_calculated_grasp_matrix = None # Stores the raw GraspNet matrix of the BEST grasp
        self.last_sorted_grasp_group = None # Stores the GraspGroup of filtered, sorted grasps
        self.camera = None
        self.net = None
        self.tf_publish_timer = None
        self.last_calculated_grasp_matrix = None # Stores the raw GraspNet matrix
        self.last_grasp_obj = None
        self.last_cloud = None
        self.visualization_enabled = cfgs.visualize
        self.r_first_offset = None # Will hold the SciPy Rotation object for the first offset
        self.r_second_offset = None # Will hold the SciPy Rotation object for the second offset
        

        # Pre-compute the fixed rotation objects
        valid_offsets = True
        try:
            self.r_first_offset = R.from_quat(self.FIRST_ROTATION_OFFSET_QUAT)
            self.get_logger().info(f"First fixed rotation offset quat defined: {self.FIRST_ROTATION_OFFSET_QUAT}")
            self.get_logger().info(f"Equivalent First Rotation Matrix:\n{self.r_first_offset.as_matrix()}")
        except ValueError as e:
             self.get_logger().error(f"Invalid first fixed rotation quat defined: {self.FIRST_ROTATION_OFFSET_QUAT} - {e}")
             self.r_first_offset = R.identity() # Use identity if invalid
             self.get_logger().warn("Using identity rotation as FIRST offset due to error.")
             valid_offsets = False

        try:
            self.r_second_offset = R.from_quat(self.SECOND_ROTATION_OFFSET_QUAT)
            self.get_logger().info(f"Second fixed rotation offset quat defined: {self.SECOND_ROTATION_OFFSET_QUAT}")
            self.get_logger().info(f"Equivalent Second Rotation Matrix:\n{self.r_second_offset.as_matrix()}")
        except ValueError as e:
             self.get_logger().error(f"Invalid second fixed rotation quat defined: {self.SECOND_ROTATION_OFFSET_QUAT} - {e}")
             self.r_second_offset = R.identity() # Use identity if invalid
             self.get_logger().warn("Using identity rotation as SECOND offset due to error.")
             valid_offsets = False

        if not valid_offsets:
            self.get_logger().warn("One or both fixed rotation offsets were invalid. TF will reflect only valid offsets.")


        try:
            # Init Camera Object
            self.get_logger().info("Initializing Camera Object...")
            self.camera = CameraRealsense(serial_number=self.cfgs.serial_number, logger=self.get_logger())

            # Init GraspNet model
            self.get_logger().info("Initializing GraspNet model...")
            self.net = self._get_net()

            # Create Service Server
            self.trigger_service = self.create_service(
                Trigger,
                'trigger_grasp_calculation',
                self.handle_trigger_request,
                callback_group=self.callback_group
            )
            self.get_logger().info(f"Service '{self.trigger_service.srv_name}' ready.")

            # --- Start TF timer - calls the function WITH the combined offsets ---
            # self.tf_publish_timer = self.create_timer(
            #     0.1, # 10 Hz
            #     self.publish_modified_grasp_tf, # Still calls the same function, logic is inside
            #     callback_group=self.callback_group
            # )
            self.get_logger().info("Started TF publishing timer (publishing MODIFIED GraspNet pose with chained offsets).")
            # ---

            self.get_logger().info("GraspNet Service Node initialized and waiting for trigger.")

        except (FileNotFoundError, Exception) as e:
            self.get_logger().fatal(f"Initialization failed: {e}")
            traceback.print_exc()
            self._cleanup_and_exit()

    # --- Cleanup and Destroy Methods ---
    def _cleanup_and_exit(self):
        # ... (implementation as before) ...
        if rclpy.ok():
            if hasattr(self, '_Node__handle') and self._Node__handle is not None: pass
            else: self.get_logger().warn("Node handle does not exist or is invalid during cleanup.")
            try: context = self.context if hasattr(self, 'context') else None; rclpy.shutdown(context=context); self.get_logger().info("RCLPY shutdown initiated.")
            except Exception as shutdown_e: self.get_logger().error(f"Error during rclpy shutdown: {shutdown_e}")
        sys.exit(1)

    def destroy_node(self):
        # ... (implementation as before) ...
        self.get_logger().info("Destroying node...")
        if self.tf_publish_timer is not None and not self.tf_publish_timer.is_canceled(): self.tf_publish_timer.cancel()
        if hasattr(self, 'camera') and self.camera: self.camera.disconnect()
        try: super().destroy_node(); self.get_logger().info("Node destruction complete (super call successful).")
        except Exception as e: self.get_logger().error(f"Error during super().destroy_node(): {e}")

    # --- Service Callback ---
    def handle_trigger_request(self, request, response):
        self.get_logger().info("Received trigger request. Starting grasp calculation...")
        grasp_matrix_raw = None
        sorted_grasp_group = None # Changed from grasp_obj_raw
        cloud = None
        start_time = time.time()

        # --- Camera Connection Logic (remains the same) ---
        if self.camera.connected: self.get_logger().warn("Camera was already connected, disconnecting first."); self.camera.disconnect(); time.sleep(0.5)
        connect_success = self.camera.connect(rgb_resolution=(self.cfgs.width, self.cfgs.height), fps=self.cfgs.fps)
        if not connect_success:
            self.get_logger().error("Failed to connect to the camera."); response.success = False; response.message = "Failed to connect to camera."
            if self.camera.connected: self.camera.disconnect()
            return response
        # --- End Camera Connection ---

        try:
            self.get_logger().info("Calculating grasps...")
            # Updated call signature
            grasp_matrix_raw, sorted_grasp_group, cloud = self._calculate_best_grasp()

            # Check if we got a valid group (even if the best matrix failed)
            if sorted_grasp_group is not None and len(sorted_grasp_group) > 0:
                num_found = len(sorted_grasp_group)
                self.get_logger().info(f"Grasp calculation successful. Found {num_found} valid grasps (after filtering/sorting).")
                self.last_calculated_grasp_matrix = grasp_matrix_raw # Store BEST matrix (might be None if best failed)
                self.last_sorted_grasp_group = sorted_grasp_group # Store the whole sorted group
                self.last_cloud = cloud
                self.last_grasp_obj = sorted_grasp_group[0] # Still store the best object if needed elsewhere

                response.success = True
                # Update message slightly
                tf_status = "will publish modified pose of BEST grasp." if grasp_matrix_raw is not None else "failed to get BEST grasp matrix for TF."
                response.message = f"Found {num_found} valid grasps. TF {tf_status} Visualization shows top 50."
                if grasp_matrix_raw is not None:
                    self.publish_modified_grasp_tf()

                # --- Visualization Section (MODIFIED) ---
                if self.visualization_enabled and self.last_sorted_grasp_group and self.last_cloud and self.last_cloud.has_points():
                    self.get_logger().info("Attempting blocking visualization (showing top 50 valid grasps)...")
                    try:
                        geoms = [self.last_cloud]
                        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                        geoms.append(cam_frame)

                        # Determine how many grasps to show (max 50)
                        num_grasps_to_show = min(len(self.last_sorted_grasp_group), 50)
                        self.get_logger().info(f"Visualizing the top {num_grasps_to_show} grasps.")

                        # Add geometries for the top N grasps
                        for i in range(num_grasps_to_show):
                            grasp = self.last_sorted_grasp_group[i]
                            # Add color based on score? Optional. Default color is fine.
                            # color = [grasp.score, 0, 1 - grasp.score] # Example: Blue to Red based on score
                            gripper_geom = grasp.to_open3d_geometry() # Use default color for now
                            if gripper_geom:
                                geoms.append(gripper_geom)
                            else:
                                self.get_logger().warn(f"Could not generate gripper geometry for grasp index {i} (score: {grasp.score:.3f}).")

                        o3d.visualization.draw_geometries(geoms,
                                                            window_name=f"GraspNet Results (Top {num_grasps_to_show} Grasps - Close to continue)",
                                                            width=1280, height=720)
                        self.get_logger().info("Visualization window closed.")
                    except Exception as vis_e:
                        self.get_logger().error(f"Error during visualization: {vis_e}")
                        traceback.print_exc()
                elif self.visualization_enabled:
                    self.get_logger().warn("Visualization enabled but required grasp group or cloud missing/empty.")
                # --- End Visualization Section ---

            else: # Handle failure case (no valid grasps found)
                self.get_logger().error("Failed to calculate grasp (no valid grasps found).")
                self.last_calculated_grasp_matrix = None
                self.last_sorted_grasp_group = None # Clear the group too
                self.last_grasp_obj = None
                self.last_cloud = None
                response.success = False
                response.message = "Failed to find any valid grasps."

        except Exception as e:
            self.get_logger().error(f"Exception during grasp calculation pipeline: {e}"); traceback.print_exc()
            self.last_calculated_grasp_matrix = None
            self.last_sorted_grasp_group = None # Clear group on exception
            self.last_grasp_obj = None
            self.last_cloud = None
            response.success = False; response.message = f"Exception during grasp calculation: {e}"
        finally:
            if self.camera.connected: self.get_logger().info("Disconnecting camera..."); self.camera.disconnect()
            else: self.get_logger().warn("Camera was not connected at the end of handle_trigger_request.")
            end_time = time.time(); self.get_logger().info(f"Grasp trigger request processed in {end_time - start_time:.2f} seconds.")
        return response
    # --- Grasp Calculation Methods ---
    # _get_net, _get_and_process_data, _get_grasps, _collision_detection, _calculate_best_grasp
    # remain unchanged as they deal with the RAW pose calculation.
    def _get_net(self):
        # ... (implementation as before) ...
        net = GraspNet(input_feature_dim=0,
                        num_view=self.cfgs.num_view, 
                        num_angle=12, num_depth=4, 
                        cylinder_radius=0.05, 
                        hmin=-0.02, 
                        hmax_list=[0.01,0.02,0.03,0.04], 
                        is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); self.get_logger().info(f"Using device: {device}"); net.to(device)
        try: checkpoint = torch.load(self.cfgs.checkpoint_path, map_location=device); net.load_state_dict(checkpoint['model_state_dict']); start_epoch = checkpoint.get('epoch', -1); self.get_logger().info("-> Loaded checkpoint %s (epoch %d)"%(self.cfgs.checkpoint_path, start_epoch))
        except FileNotFoundError as e: self.get_logger().error(f"Checkpoint file not found: {self.cfgs.checkpoint_path}"); raise e
        except KeyError as e: self.get_logger().error(f"Error loading checkpoint: Missing key {e}. Checkpoint might be corrupted or from a different version."); raise e
        except Exception as e: self.get_logger().error(f"Error loading checkpoint: {e}"); traceback.print_exc(); raise e
        net.eval(); return net

    def _get_and_process_data(self):
        # ... (implementation as before) ...
        self.get_logger().debug("Capture single frame...");
        if not self.camera or not self.camera.connected: self.get_logger().error("Camera not connected for data capture."); return None, None
        color_bgr, depth_m = self.camera.get_rgbd()
        if color_bgr is None or depth_m is None: self.get_logger().warn("Failed to get RGB-D frame."); return None, None
        if color_bgr.size == 0 or depth_m.size == 0: self.get_logger().warn("Received empty color or depth image."); return None, None
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB); color_norm = color_rgb.astype(np.float32) / 255.0
        try: intrinsic_matrix = self.camera.camera_k()
        except Exception as e: self.get_logger().error(f"Failed to get camera intrinsics: {e}"); return None, None
        if intrinsic_matrix is None: self.get_logger().error("Camera intrinsics are None."); return None, None # Added check
        width=color_bgr.shape[1]; height=color_bgr.shape[0]; k=intrinsic_matrix; fx,fy,cx,cy=k[0,0],k[1,1],k[0,2],k[1,2]; factor_depth=1.0
        self.get_logger().debug("Generating point cloud...");
        if np.all(depth_m == 0): self.get_logger().warn("Depth image is all zeros."); return None, None
        cam_info = CameraInfo(float(width),float(height),fx,fy,cx,cy,factor_depth)
        cloud = create_point_cloud_from_depth_image(depth_m, cam_info, organized=True)
        if cloud is None: self.get_logger().error("Failed to create point cloud."); return None, None
        depth_min=self.cfgs.depth_min; depth_max=self.cfgs.depth_max
        valid_depth_mask = (depth_m > depth_min) & (depth_m < depth_max)
        cloud_flat=cloud.reshape(-1,3); color_flat=color_norm.reshape(-1,3); depth_mask_flat=valid_depth_mask.reshape(-1)
        valid_points_mask = np.isfinite(cloud_flat).all(axis=1); final_mask = depth_mask_flat & valid_points_mask
        cloud_masked = cloud_flat[final_mask]; color_masked = color_flat[final_mask]
        num_valid_points = len(cloud_masked)
        self.get_logger().debug(f"Number of valid points after depth/NaN filtering: {num_valid_points}")
        if num_valid_points < self.cfgs.num_point * 0.1: self.get_logger().warn(f"Too few valid points ({num_valid_points}) after filtering! Min required approx: {self.cfgs.num_point * 0.1}. Check depth range/camera view."); return None, None
        n_sample=self.cfgs.num_point; self.get_logger().debug(f"Sampling {n_sample} points...");
        if num_valid_points >= n_sample: idxs = np.random.choice(num_valid_points, n_sample, replace=False)
        else: self.get_logger().warn(f"Sampling {n_sample} points with replacement from {num_valid_points} available points."); idxs = np.random.choice(num_valid_points, n_sample, replace=True)
        cloud_sampled = cloud_masked[idxs]
        end_pts={}; cloud_torch = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); cloud_torch = cloud_torch.to(dev)
        end_pts['point_clouds'] = cloud_torch
        collision_cloud = o3d.geometry.PointCloud(); collision_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32)); collision_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        self.get_logger().debug(f"Processed data successfully. Sampled cloud shape: {cloud_sampled.shape}, Collision cloud points: {len(collision_cloud.points)}")
        return end_pts, collision_cloud

    def _get_grasps(self, end_points):
        # ... (implementation as before) ...
        self.get_logger().debug("Inferring grasps from sampled point cloud...");
        if end_points is None or 'point_clouds' not in end_points or end_points['point_clouds'].nelement() == 0: self.get_logger().error("Input end_points for grasp inference is invalid or empty."); return GraspGroup()
        try:
            with torch.no_grad(): pred = self.net(end_points); grasps = pred_decode(pred)
            if grasps[0] is None: self.get_logger().warn("GraspNet prediction returned None."); return GraspGroup()
            gg_arr = grasps[0].detach().cpu().numpy()
            if gg_arr.size == 0: self.get_logger().warn("GraspNet prediction returned an empty grasp array."); return GraspGroup()
            gg = GraspGroup(gg_arr); self.get_logger().debug(f"Network predicted {len(gg)} raw grasps."); return gg
        except Exception as e: self.get_logger().error(f"Exception during grasp inference or decoding: {e}"); traceback.print_exc(); return GraspGroup()

    def _collision_detection(self, gg, cloud_o3d):
        # ... (implementation as before) ...
        num_grasps_before = len(gg)
        if num_grasps_before == 0: return gg
        if not cloud_o3d or not cloud_o3d.has_points(): self.get_logger().warn("Collision cloud is empty or invalid. Skipping collision detection."); return gg
        self.get_logger().debug(f"Performing collision detection for {num_grasps_before} grasps...");
        pts=np.asarray(cloud_o3d.points);
        if pts.shape[0]==0: self.get_logger().warn("Collision cloud points array is empty. Skipping collision detection."); return gg
        try:
            detector = ModelFreeCollisionDetector(pts, voxel_size=self.cfgs.voxel_size)
            mask, _ = detector.detect(gg, approach_dist=0.01, collision_thresh=self.cfgs.collision_thresh)
            n_coll = np.sum(mask); n_valid = num_grasps_before - n_coll
            gg_filt = gg[~mask]; self.get_logger().debug(f"Collision check complete: {n_coll} collisions detected, {n_valid} grasps remaining."); return gg_filt
        except IndexError as e: self.get_logger().error(f"IndexError during collision detection: {e}. Check grasp group and points."); traceback.print_exc(); return GraspGroup()
        except Exception as e: self.get_logger().error(f"Unexpected error during collision detection: {e}"); self.get_logger().warn("Returning grasps without collision check due to error."); traceback.print_exc(); return gg

        # Change the return type hint for clarity
    def _calculate_best_grasp(self) -> Tuple[Optional[np.ndarray], Optional[GraspGroup], Optional[o3d.geometry.PointCloud]]:
        """
        Calculates grasps, filters collisions, sorts them, and returns:
        - The 4x4 transformation matrix of the *best* grasp (for TF).
        - The *entire sorted GraspGroup* object containing valid grasps.
        - The point cloud used for collision detection.
        """
        t_start = time.time()
        T_graspnet_best_raw = None # Matrix of the single best grasp
        sorted_filtered_gg = None  # The group of filtered, sorted grasps
        cloud = None

        try:
            # 1. Acquire and process data (remains the same)
            t_d0 = time.time()
            end_pts, cloud = self._get_and_process_data()
            t_d1 = time.time()
            self.get_logger().debug(f"Data acquisition and processing: {t_d1 - t_d0:.4f}s")
            if end_pts is None or cloud is None:
                self.get_logger().warn("Failed to get valid data/point cloud.")
                return None, None, None

            # 2. Get grasps (remains the same)
            t_g0 = time.time()
            gg = self._get_grasps(end_pts)
            t_g1 = time.time()
            self.get_logger().debug(f"Grasp inference: {t_g1 - t_g0:.4f}s")
            if len(gg) == 0:
                self.get_logger().warn("No grasps predicted by the network.")
                return None, None, cloud # Return None for matrix and group

            # 3. Collision detection (remains the same)
            t_c0 = time.time()
            if self.cfgs.collision_thresh > 0:
                gg_f = self._collision_detection(gg, cloud)
            else:
                self.get_logger().info("Collision threshold is 0, skipping collision detection.")
                gg_f = gg
            t_c1 = time.time()
            self.get_logger().debug(f"Collision detection: {t_c1 - t_c0:.4f}s")
            if len(gg_f) == 0:
                self.get_logger().warn("No grasps remaining after collision filtering.")
                return None, None, cloud # Return None for matrix and group

            # 4. Sort by score
            t_s0 = time.time()
            gg_f.sort_by_score() # Sorts the group in place
            sorted_filtered_gg = gg_f # Store the sorted group
            t_s1 = time.time()
            self.get_logger().debug(f"Grasp sorting: {t_s1 - t_s0:.4f}s")

            # 5. Get the *best* grasp details for TF and logging
            if len(sorted_filtered_gg) > 0:
                best_grasp_raw = sorted_filtered_gg[0] # Get the top one from the sorted group
                self.get_logger().info(f"Selected BEST grasp (for TF) -> Score:{best_grasp_raw.score:.4f}, "
                                       f"Width:{best_grasp_raw.width:.4f}m, Depth:{best_grasp_raw.depth:.4f}m")

                if hasattr(best_grasp_raw, 'rotation_matrix') and hasattr(best_grasp_raw, 'translation'):
                    T_graspnet_best_raw = np.eye(4)
                    T_graspnet_best_raw[:3, :3] = best_grasp_raw.rotation_matrix
                    T_graspnet_best_raw[:3, 3] = best_grasp_raw.translation

                    if not np.isfinite(T_graspnet_best_raw).all():
                        self.get_logger().error("Best grasp resulted in a non-finite RAW transformation matrix! Cannot use for TF.")
                        T_graspnet_best_raw = None # Nullify if invalid
                    else:
                         # Log the raw best grasp details (optional, keep if useful)
                        self.get_logger().info("===== Best Grasp (raw from GraspNet, used for TF) =====")
                        # ... (keep logging details if needed) ...
                        self.get_logger().info(f"4x4 Homogeneous Transform (T_graspnet_best_raw):\n{T_graspnet_best_raw}")
                        self.get_logger().info("===========================================")
                else:
                    self.get_logger().error("Selected best grasp object is missing 'rotation_matrix' or 'translation' attributes. Cannot generate TF matrix.")
                    T_graspnet_best_raw = None
            else:
                 self.get_logger().warn("Sorted grasp group is empty after all steps, cannot determine best grasp matrix.")
                 T_graspnet_best_raw = None


            t_end = time.time()
            self.get_logger().info(f"-> Grasp calculation sub-pipeline finished: {t_end - t_start:.4f}s. Found {len(sorted_filtered_gg)} valid grasps.")
            # Return the BEST matrix, the SORTED GROUP, and the cloud
            return T_graspnet_best_raw, sorted_filtered_gg, cloud

        except Exception as e:
            self.get_logger().error(f"Exception in _calculate_best_grasp: {e}")
            traceback.print_exc()
            return None, None, cloud # Return None for matrix and group on error

    # --- TF publishing function (PUBLISHING MODIFIED POSE WITH CHAINED OFFSETS) ---
    def publish_modified_grasp_tf(self):
        """
        Applies two fixed rotation offsets sequentially to the raw GraspNet
        calculated grasp pose and publishes the resulting final modified pose as TF.
        """
        if self.last_calculated_grasp_matrix is None or self.r_first_offset is None or self.r_second_offset is None:
            # Don't publish if no grasp or any offset is invalid
            if self.last_calculated_grasp_matrix is not None and (self.r_first_offset is None or self.r_second_offset is None):
                 self.get_logger().warn("Cannot publish TF: One or both fixed rotation offsets are invalid/None.")
            return
        


        T_graspnet_raw = self.last_calculated_grasp_matrix

        try:
            # 1. Extract raw rotation matrix and translation vector
            R_graspnet_matrix_raw = T_graspnet_raw[:3, :3]
            t_graspnet_vec_raw = T_graspnet_raw[:3, 3] # Translation remains the same

            if not np.isfinite(R_graspnet_matrix_raw).all() or not np.isfinite(t_graspnet_vec_raw).all():
                self.get_logger().warn("Stored raw grasp matrix has non-finite values. Skipping TF publish.")
                self.last_calculated_grasp_matrix = None # Clear invalid matrix
                return

            # 2. Convert raw rotation matrix to SciPy Rotation object
            r_graspnet_raw = R.from_matrix(R_graspnet_matrix_raw)

            # 3. Apply the fixed rotation offsets sequentially
            # Order: r_raw * r_offset1 * r_offset2
            r_modified = r_graspnet_raw * self.r_first_offset * self.r_second_offset

            # 4. Get the quaternion from the FINAL MODIFIED rotation
            quat_modified = r_modified.as_quat() # [x, y, z, w] for ROS

            # --- Logging Modified Pose ---
            self.get_logger().info(f"--- Publishing MODIFIED Grasp TF (Chained Offsets) ---")
            self.get_logger().info(f"  Parent Frame: {self.cfgs.camera_link}")
            self.get_logger().info(f"  Child Frame:  estimated_grasp")
            self.get_logger().info(f"  Raw Translation: [{t_graspnet_vec_raw[0]:.3f}, {t_graspnet_vec_raw[1]:.3f}, {t_graspnet_vec_raw[2]:.3f}] (Published)")
            self.get_logger().info(f"  Raw Quat (xyzw): [{r_graspnet_raw.as_quat()[0]:.3f}, {r_graspnet_raw.as_quat()[1]:.3f}, {r_graspnet_raw.as_quat()[2]:.3f}, {r_graspnet_raw.as_quat()[3]:.3f}]")
            self.get_logger().info(f"  Offset1 Quat:    [{self.r_first_offset.as_quat()[0]:.3f}, {self.r_first_offset.as_quat()[1]:.3f}, {self.r_first_offset.as_quat()[2]:.3f}, {self.r_first_offset.as_quat()[3]:.3f}]")
            self.get_logger().info(f"  Offset2 Quat:    [{self.r_second_offset.as_quat()[0]:.3f}, {self.r_second_offset.as_quat()[1]:.3f}, {self.r_second_offset.as_quat()[2]:.3f}, {self.r_second_offset.as_quat()[3]:.3f}]")
            self.get_logger().info(f"  FINAL MODIFIED Quat (xyzw): [{quat_modified[0]:.3f}, {quat_modified[1]:.3f}, {quat_modified[2]:.3f}, {quat_modified[3]:.3f}] (Published)")
            # self.get_logger().debug(f"Final Modified Rot Matrix:\n{r_modified.as_matrix()}") # Optional debug
            self.get_logger().info(f"------")

            # --- Publish the FINAL MODIFIED pose ---
            now = self.get_clock().now().to_msg()
            t_stamped = TransformStamped()
            t_stamped.header.stamp = now
            t_stamped.header.frame_id = self.cfgs.camera_link # Parent is optical frame
            t_stamped.child_frame_id = 'estimated_grasp'     # Child is the grasp frame

            # Translation IS the original raw translation
            t_stamped.transform.translation.x = t_graspnet_vec_raw[0]
            t_stamped.transform.translation.y = t_graspnet_vec_raw[1]
            t_stamped.transform.translation.z = t_graspnet_vec_raw[2]

            # Rotation IS the FINAL MODIFIED quaternion
            t_stamped.transform.rotation.x = quat_modified[0]
            t_stamped.transform.rotation.y = quat_modified[1]
            t_stamped.transform.rotation.z = quat_modified[2]
            t_stamped.transform.rotation.w = quat_modified[3]

            self.tf_broadcaster.sendTransform(t_stamped)
            self.get_logger().info("Modified grasp TF published.")

        except Exception as e:
             self.get_logger().error(f"Failed during final modified TF publishing: {e}")
             traceback.print_exc()
             # Optionally clear the matrix if publishing fails consistently
             # self.last_calculated_grasp_matrix = None


# ... (Rest of the script, including CameraRealsense and main function, remains the same as the previous version) ...
# Ensure the main function is present and correct as in the previous answer.

# =============================================================================
# RealSense Camera Class (Using Reverted Settings) - Keep as before
# =============================================================================
class CameraRealsense:
    # ... (exact implementation from previous answer) ...
    def __init__(self, serial_number: str=None, logger=None):
        self._logger = logger if logger else logging.getLogger(self.__class__.__name__)
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._pipeline_profile = None
        self._align = None
        self._intrinsics = None
        self._depth_scale = None
        self._connected = False
        self._serial_number = serial_number
        self._rgb_resolution = (1280, 720)
        self._depth_resolution = (1280, 720)
        self._fps = 15

    def connect(self, rgb_resolution: Tuple[int, int]=(1280, 720), depth_resolution: Tuple[int, int]=(1280, 720), fps: int=15):
        if self._connected:
            self._logger.warning("Camera already connected.")
            return True
        self._rgb_resolution=rgb_resolution
        self._depth_resolution=depth_resolution
        self._fps=fps
        if self._depth_resolution != self._rgb_resolution:
            self._logger.warning(f"Depth resolution {self._depth_resolution} differs from RGB {self._rgb_resolution}. Forcing depth to match RGB.")
            self._depth_resolution = self._rgb_resolution
        self._logger.info(f"Attempting to connect RealSense camera...")
        self._logger.info(f"  Requested RGB resolution: {self._rgb_resolution}")
        self._logger.info(f"  Requested Depth resolution: {self._depth_resolution}")
        self._logger.info(f"  Requested FPS: {self._fps}")
        if self._serial_number:
            self._logger.info(f"  Targeting serial number: {self._serial_number}")
            self._config.enable_device(self._serial_number)
        else: self._logger.info("  No specific serial number targeted.")
        try:
            self._config.enable_stream(rs.stream.depth, *self._depth_resolution, rs.format.z16, self._fps)
            self._config.enable_stream(rs.stream.color, *self._rgb_resolution, rs.format.bgr8, self._fps)
            pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
            self._pipeline_profile = self._config.resolve(pipeline_wrapper)
            device = self._pipeline_profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            actual_serial = device.get_info(rs.camera_info.serial_number)
            self._logger.info(f"Successfully resolved configuration for device: {device_name} (Serial: {actual_serial})")
            depth_sensor = device.first_depth_sensor()
            if not depth_sensor: self._logger.error("Could not get depth sensor from device."); return False
            self._depth_scale = depth_sensor.get_depth_scale()
            self._logger.info(f"Depth scale: {self._depth_scale}")
            self._logger.info(f"Checking if specific settings are needed for: {device_name}")
            if "D435" in device_name or "D435I" in device_name:
                self._apply_d435_settings(device, depth_sensor) # Uses the reverted fixed settings method below
            else: self._logger.info(f"No specific settings applied for {device_name}.")
            self._profile = self._pipeline.start(self._config)
            self._logger.info("Pipeline started.")
            self._align_to = rs.stream.color
            self._align = rs.align(self._align_to)
            self._logger.info(f"Alignment configured to stream: {self._align_to}")
            self._logger.info("Allowing camera stream to stabilize...")
            time.sleep(1.5)
            color_profile = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
            if not color_profile: self._logger.error("Could not get color profile after starting pipeline."); self.disconnect(); return False
            self._intrinsics = color_profile.get_intrinsics()
            self._logger.info(f"Intrinsics obtained: fx={self._intrinsics.fx}, fy={self._intrinsics.fy}, ppx={self._intrinsics.ppx}, ppy={self._intrinsics.ppy}")
            self._connected = True
            self._logger.info("RealSense camera connected and configured successfully.")
            return True
        except RuntimeError as e: self._logger.error(f"RuntimeError during camera connection: {e}"); traceback.print_exc(); self._connected = False; return False
        except Exception as e: self._logger.error(f"Unexpected error during camera connection: {e}"); traceback.print_exc(); self._connected = False; return False

    # --- REVERTED D435 FIXED SETTINGS METHOD ---
    def _apply_d435_settings(self, device, depth_sensor):
        """Applies specific fixed settings for D435/D435i cameras."""
        self._logger.info("Applying specific D435(i) options (fixed settings)...")
        try: # Stereo Module Settings
            stereo_module = next((s for s in device.query_sensors() if s.get_info(rs.camera_info.name) == 'Stereo Module'), None)
            if stereo_module:
                opt_emitter = rs.option.emitter_enabled
                if stereo_module.supports(opt_emitter) and not stereo_module.is_option_read_only(opt_emitter): stereo_module.set_option(opt_emitter, 1.0); self._logger.info("  Stereo Module: Emitter Enabled set to 1 (ON)")
                else: self._logger.warn("  Stereo Module: Cannot set Emitter Enabled.")
                opt_laser = rs.option.laser_power
                if stereo_module.supports(opt_laser) and not stereo_module.is_option_read_only(opt_laser):
                    laser_range = stereo_module.get_option_range(opt_laser); set_laser = min(360.0, laser_range.max)
                    stereo_module.set_option(opt_laser, set_laser); self._logger.info(f"  Stereo Module: Laser Power set to {set_laser} (Max possible: {laser_range.max})")
                else: self._logger.warn("  Stereo Module: Cannot set Laser Power.")
            else: self._logger.warn("Stereo Module not found on device.")
            # Depth Sensor Manual Exposure/Gain
            opt_depth_ae = rs.option.enable_auto_exposure
            if depth_sensor.supports(opt_depth_ae) and not depth_sensor.is_option_read_only(opt_depth_ae):
                 depth_sensor.set_option(opt_depth_ae, 0); self._logger.info("  Depth Sensor: Auto Exposure Disabled.")
                 opt_depth_exp = rs.option.exposure
                 if depth_sensor.supports(opt_depth_exp) and not depth_sensor.is_option_read_only(opt_depth_exp): depth_sensor.set_option(opt_depth_exp, 750.0); self._logger.info(f"  Depth Sensor: Manual Exposure set to 750.0")
                 else: self._logger.warn("  Depth Sensor: Cannot set Manual Exposure.")
                 opt_depth_gain = rs.option.gain
                 if depth_sensor.supports(opt_depth_gain) and not depth_sensor.is_option_read_only(opt_depth_gain): depth_sensor.set_option(opt_depth_gain, 16.0); self._logger.info(f"  Depth Sensor: Manual Gain set to 16.0")
                 else: self._logger.warn("  Depth Sensor: Cannot set Manual Gain.")
            else: self._logger.warn("  Depth Sensor: Cannot disable Auto Exposure to set manual values.")
        except Exception as e: self._logger.warn(f"Could not set some depth/stereo options: {e}")
        try: # RGB Sensor Manual Exposure/Gain/White Balance
            color_sensor = device.first_color_sensor()
            if color_sensor:
                opt_rgb_ae = rs.option.enable_auto_exposure
                if color_sensor.supports(opt_rgb_ae) and not color_sensor.is_option_read_only(opt_rgb_ae):
                    color_sensor.set_option(opt_rgb_ae, 0); self._logger.info("  RGB Sensor: Auto Exposure Disabled.")
                    opt_rgb_exp = rs.option.exposure
                    if color_sensor.supports(opt_rgb_exp) and not color_sensor.is_option_read_only(opt_rgb_exp): color_sensor.set_option(opt_rgb_exp, 500.0); self._logger.info(f"  RGB Sensor: Manual Exposure set to 500.0")
                    else: self._logger.warn("  RGB Sensor: Cannot set Manual Exposure.")
                    opt_rgb_gain = rs.option.gain
                    if color_sensor.supports(opt_rgb_gain) and not color_sensor.is_option_read_only(opt_rgb_gain): color_sensor.set_option(opt_rgb_gain, 16.0); self._logger.info(f"  RGB Sensor: Manual Gain set to 16.0")
                    else: self._logger.warn("  RGB Sensor: Cannot set Manual Gain.")
                else: self._logger.warn("  RGB Sensor: Cannot disable Auto Exposure to set manual values.")
                opt_rgb_awb = rs.option.enable_auto_white_balance
                if color_sensor.supports(opt_rgb_awb) and not color_sensor.is_option_read_only(opt_rgb_awb):
                     color_sensor.set_option(opt_rgb_awb, 0); self._logger.info("  RGB Sensor: Auto White Balance Disabled.")
                     opt_rgb_wb = rs.option.white_balance
                     if color_sensor.supports(opt_rgb_wb) and not color_sensor.is_option_read_only(opt_rgb_wb): color_sensor.set_option(opt_rgb_wb, 5500.0); self._logger.info(f"  RGB Sensor: Manual White Balance set to 5500.0")
                     else: self._logger.warn("  RGB Sensor: Cannot set Manual White Balance.")
                else: self._logger.warn("  RGB Sensor: Cannot disable Auto White Balance to set manual value.")
            else: self._logger.warn("RGB color sensor not found on device.")
        except Exception as e: self._logger.warn(f"Could not set some RGB options: {e}")
    # --- END REVERTED METHOD ---

    def camera_k(self) -> Optional[np.ndarray]:
        # ... (implementation as before) ...
        if not self._connected or not self._intrinsics: self._logger.error("Camera not connected or intrinsics unavailable."); return None
        k = self._intrinsics; return np.array([[k.fx, 0, k.ppx], [0, k.fy, k.ppy], [0, 0, 1]], dtype=np.float32)

    def get_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # ... (implementation as before) ...
        if not self._connected: self._logger.error("Cannot get RGBD frame, camera not connected."); return None, None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=5000);
            if not frames: self._logger.warn("No frames received within timeout period (5000 ms)."); return None, None
            aligned_frames = self._align.process(frames)
            if not aligned_frames: self._logger.warn("Frame alignment failed."); return None, None
            aligned_depth_frame = aligned_frames.get_depth_frame(); color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame: self._logger.warn("Could not retrieve aligned depth or color frame after alignment."); return None, None
            depth_raw = np.asanyarray(aligned_depth_frame.get_data()); depth_m = depth_raw.astype(np.float32) * self._depth_scale
            color_bgr = np.asanyarray(color_frame.get_data())
            if depth_m.shape[:2] != color_bgr.shape[:2]: self._logger.error(f"Aligned frame shape mismatch! Depth: {depth_m.shape}, Color: {color_bgr.shape}"); return None, None
            return color_bgr, depth_m
        except RuntimeError as e: self._logger.error(f"RuntimeError while waiting for frames: {e}"); return None, None
        except Exception as e: self._logger.error(f"Unexpected error in get_rgbd: {e}"); traceback.print_exc(); return None, None

    def disconnect(self):
        # ... (implementation as before) ...
        if self._connected:
            self._logger.info("Disconnecting RealSense camera...");
            try: self._pipeline.stop(); self._connected = False; self._intrinsics = None; self._profile = None; self._pipeline_profile = None; self._logger.info("RealSense camera disconnected.")
            except Exception as e: self._logger.error(f"Error stopping pipeline: {e}"); self._connected = False
        else: self._logger.info("Camera already disconnected.")

    @property
    def pipeline(self): return self._pipeline
    @property
    def align(self): return self._align
    @property
    def connected(self): return self._connected
    def __del__(self): self.disconnect()


def main(args=None):
    # ... (exact implementation from previous answer) ...
    rclpy.init(args=args)
    parser = argparse.ArgumentParser("GraspNet ROS 2 Service Node")
    # Arguments
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--num_point', type=int, default=40000, help="Number of points to sample from the point cloud for GraspNet input.")
    parser.add_argument('--num_view', type=int, default=300, help="Number of views used during GraspNet training (parameter for the model).")
    parser.add_argument('--collision_thresh', type=float, default=0.01, help="Collision threshold distance [m]. Set to 0 to disable collision checking.")
    parser.add_argument('--voxel_size', type=float, default=0.005, help="Voxel size [m] for collision detection environment.")
    parser.add_argument('--serial_number', type=str, default=None, help="Specific serial number of the RealSense camera to use.")
    parser.add_argument('--width', type=int, default=1280, help="Camera image width.")
    parser.add_argument('--height', type=int, default=720, help="Camera image height.")
    parser.add_argument('--fps', type=int, default=15, help="Camera FPS.")
    parser.add_argument('--depth_min', type=float, default=0.3, help="Minimum depth range [m] to consider.")
    parser.add_argument('--depth_max', type=float, default=0.6, help="Maximum depth range [m] to consider.")
    parser.add_argument('--camera_link', type=str, default='camera_link', # Common optical frame name
                        help='TF parent frame for the estimated_grasp. MUST be the optical frame of the camera used by GraspNet. [default: camera_link]')
    parser.add_argument('--visualize', action='store_true', help="Show Open3D visualization of the point cloud and the selected RAW grasp pose.")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'], help='Set the logging level.')
    cfgs, _ = parser.parse_known_args()
    log_level_numeric = getattr(logging, cfgs.log_level.upper(), logging.INFO)
    # Configure ROS 2 logging level
    log_level_ros = getattr(rclpy.logging.LoggingSeverity, cfgs.log_level.upper(), rclpy.logging.LoggingSeverity.INFO)
    rclpy.logging.set_logger_level('graspnet_service_node', log_level_ros) # Set for the node's logger
    rclpy.logging.set_logger_level('CameraRealsense', log_level_ros) # Set for the camera logger if needed
    # Configure Python standard logging (optional, but can be helpful for non-ROS libs)
    logging.basicConfig(level=log_level_numeric, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main_logger = rclpy.logging.get_logger("GraspNetNodeMain") # Use ROS logger for main setup info

    if not os.path.isfile(cfgs.checkpoint_path): main_logger.fatal(f"Checkpoint file not found: {cfgs.checkpoint_path}"); sys.exit(1)
    else: main_logger.info(f"Using checkpoint: {cfgs.checkpoint_path}")
    main_logger.info(f"Target camera frame (parent for grasp): {cfgs.camera_link}")
    main_logger.info(f"Collision check enabled: {cfgs.collision_thresh > 0}, Threshold: {cfgs.collision_thresh}, Voxel Size: {cfgs.voxel_size}")
    main_logger.info(f"Visualization: {'Enabled (shows RAW pose)' if cfgs.visualize else 'Disabled'}")
    main_logger.info(f"Depth Range: [{cfgs.depth_min}, {cfgs.depth_max}] m")

    node = None
    executor = SingleThreadedExecutor()
    try:
        node = GraspServiceNode(cfgs)
        executor.add_node(node)
        node.get_logger().info("GraspNet Service Node spinning... Waiting for trigger on '/trigger_grasp_calculation'. Ctrl+C to stop.") # Use node's logger
        executor.spin()
    except KeyboardInterrupt: main_logger.info("\nKeyboard interrupt received.")
    except Exception as e: main_logger.fatal(f"Unhandled exception in main execution loop: {e}"); traceback.print_exc()
    finally:
        main_logger.info("Initiating shutdown sequence...")
        try: executor.shutdown(); main_logger.info("Executor shutdown complete.")
        except Exception as exec_shutdown_e: main_logger.error(f"Error during executor shutdown: {exec_shutdown_e}")
        if node is not None:
             if rclpy.ok(): main_logger.info("Destroying node..."); node.destroy_node(); main_logger.info("Node destruction sequence finished.")
             else: main_logger.warn("RCLPY context already invalid, cannot explicitly destroy node.")
        # Ensure final shutdown even if node destruction failed
        if rclpy.ok():
            try: rclpy.shutdown(); main_logger.info("RCLPY shutdown complete.")
            except Exception as rclpy_shutdown_e: main_logger.error(f"Error during final RCLPY shutdown: {rclpy_shutdown_e}")
        else: main_logger.info("RCLPY context was already shut down.")
        main_logger.info("Shutdown process finished.")


if __name__ == '__main__':
    main()