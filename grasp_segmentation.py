#!/usr/bin/env python3

import os
import sys
import numpy as np
import open3d as o3d
import argparse
from scipy.spatial.transform import Rotation as R # Used for conversion AND multiplication
from scipy.spatial import KDTree
import time
import logging
import traceback

# --- Camera Imports ---
import pyrealsense2 as rs
import cv2
from typing import Tuple, Optional, List

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

# --- SciPy Import for KDTree ---
try:
    from scipy.spatial import KDTree
except ImportError:
    print("ERROR: scipy is not installed. Please install it: pip install scipy")
    sys.exit(1)


# --- GraspNet Imports ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
try:
    from graspnet import GraspNet, pred_decode
    grasp_api_path_util = os.path.join(ROOT_DIR, 'utils', 'graspnetAPI')
    grasp_api_path_root = os.path.join(ROOT_DIR, 'graspnetAPI')
    if os.path.exists(grasp_api_path_util): sys.path.append(os.path.join(ROOT_DIR, 'utils'))
    elif os.path.exists(grasp_api_path_root): sys.path.append(grasp_api_path_root)
    from graspnetAPI import GraspGroup, Grasp
    from collision_detector import ModelFreeCollisionDetector
    from data_utils import CameraInfo, create_point_cloud_from_depth_image
except ImportError as e: print(f"Import Error: {e}\nCheck GraspNet setup and PYTHONPATH.\nSys Path: {sys.path}"); sys.exit(1)
except NameError as e: print(f"NameError: {e}. Likely graspnetAPI not found. Check paths.\nSys Path: {sys.path}"); sys.exit(1)


# =============================================================================
# Deprojection Function (Keep as before)
# =============================================================================
def deproject_masked_points(mask: np.ndarray, depth_map: np.ndarray, camera_intrinsics: rs.intrinsics) -> List[Tuple[float, float, float]]:
    # ... (implementation is the same) ...
    if mask is None or depth_map is None or camera_intrinsics is None: print("[deproject_masked_points] Error: Received None input."); return []
    if mask.shape != depth_map.shape: print(f"[deproject_masked_points] Error: Mask shape {mask.shape} doesn't match depth map shape {depth_map.shape}."); return []
    if not np.any(mask): return []
    y_indices, x_indices = np.where(mask)
    depths = depth_map[y_indices, x_indices]
    points_3d = []
    for x, y, depth in zip(x_indices, y_indices, depths):
        if depth > 0:
            try:
                point_3d_raw = rs.rs2_deproject_pixel_to_point(camera_intrinsics, [float(x), float(y)], float(depth))
                points_3d.append(tuple(point_3d_raw))
            except Exception as e: print(f"[deproject_masked_points] Warning: Error deprojecting pixel ({x},{y}) with depth {depth}: {e}"); continue
    return points_3d

# =============================================================================
# RealSense Camera Class (Keep as before)
# =============================================================================
class CameraRealsense:
    # ... (implementation is the same as previous answer) ...
    def __init__(self, serial_number: str=None, logger=None):
        self._logger = logger if logger else logging.getLogger(self.__class__.__name__); self._pipeline = rs.pipeline(); self._config = rs.config()
        self._pipeline_profile = None; self._align = None; self._intrinsics = None; self._depth_scale = None; self._connected = False; self._serial_number = serial_number
        self._rgb_resolution = (1280, 720); self._depth_resolution = (1280, 720); self._fps = 15
    def connect(self, rgb_resolution: Tuple[int, int]=(1280, 720), depth_resolution: Tuple[int, int]=(1280, 720), fps: int=15):
        if self._connected: self._logger.warning("Camera already connected."); return True
        self._rgb_resolution=rgb_resolution; self._depth_resolution=depth_resolution; self._fps=fps
        if self._depth_resolution != self._rgb_resolution: self._logger.warning(f"Depth resolution {self._depth_resolution} differs from RGB {self._rgb_resolution}. Forcing depth to match RGB."); self._depth_resolution = self._rgb_resolution
        self._logger.info(f"Attempting to connect RealSense camera..."); self._logger.info(f"  Requested RGB resolution: {self._rgb_resolution}"); self._logger.info(f"  Requested Depth resolution: {self._depth_resolution}"); self._logger.info(f"  Requested FPS: {self._fps}")
        if self._serial_number: self._logger.info(f"  Targeting serial number: {self._serial_number}"); self._config.enable_device(self._serial_number)
        else: self._logger.info("  No specific serial number targeted.")
        try:
            self._config.enable_stream(rs.stream.depth, *self._depth_resolution, rs.format.z16, self._fps); self._config.enable_stream(rs.stream.color, *self._rgb_resolution, rs.format.bgr8, self._fps)
            pipeline_wrapper = rs.pipeline_wrapper(self._pipeline); self._pipeline_profile = self._config.resolve(pipeline_wrapper)
            device = self._pipeline_profile.get_device(); device_name = device.get_info(rs.camera_info.name); actual_serial = device.get_info(rs.camera_info.serial_number)
            self._logger.info(f"Successfully resolved configuration for device: {device_name} (Serial: {actual_serial})")
            depth_sensor = device.first_depth_sensor();
            if not depth_sensor: self._logger.error("Could not get depth sensor from device."); return False
            self._depth_scale = depth_sensor.get_depth_scale(); self._logger.info(f"Depth scale: {self._depth_scale}")
            self._logger.info(f"Checking if specific settings are needed for: {device_name}")
            if "D435" in device_name or "D435I" in device_name: self._apply_d435_settings(device, depth_sensor)
            else: self._logger.info(f"No specific settings applied for {device_name}.")
            self._profile = self._pipeline.start(self._config); self._logger.info("Pipeline started.")
            self._align_to = rs.stream.color; self._align = rs.align(self._align_to); self._logger.info(f"Alignment configured to stream: {self._align_to}")
            self._logger.info("Allowing camera stream to stabilize..."); time.sleep(1.5)
            color_profile = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
            if not color_profile: self._logger.error("Could not get color profile after starting pipeline."); self.disconnect(); return False
            self._intrinsics = color_profile.get_intrinsics(); self._logger.info(f"Intrinsics obtained: fx={self._intrinsics.fx}, fy={self._intrinsics.fy}, ppx={self._intrinsics.ppx}, ppy={self._intrinsics.ppy}")
            self._connected = True; self._logger.info("RealSense camera connected and configured successfully.")
            return True
        except RuntimeError as e: self._logger.error(f"RuntimeError during camera connection: {e}"); traceback.print_exc(); self._connected = False; return False
        except Exception as e: self._logger.error(f"Unexpected error during camera connection: {e}"); traceback.print_exc(); self._connected = False; return False
    def _apply_d435_settings(self, device, depth_sensor):
        # ... (implementation is the same) ...
        self._logger.info("Applying specific D435(i) options (fixed settings)...")
        try: # Stereo Module Settings
            stereo_module = next((s for s in device.query_sensors() if s.get_info(rs.camera_info.name) == 'Stereo Module'), None)
            if stereo_module:
                opt_emitter = rs.option.emitter_enabled; opt_laser = rs.option.laser_power
                if stereo_module.supports(opt_emitter) and not stereo_module.is_option_read_only(opt_emitter): stereo_module.set_option(opt_emitter, 1.0); self._logger.info("  Stereo Module: Emitter Enabled set to 1 (ON)")
                else: self._logger.warn("  Stereo Module: Cannot set Emitter Enabled.")
                if stereo_module.supports(opt_laser) and not stereo_module.is_option_read_only(opt_laser): laser_range = stereo_module.get_option_range(opt_laser); set_laser = min(360.0, laser_range.max); stereo_module.set_option(opt_laser, set_laser); self._logger.info(f"  Stereo Module: Laser Power set to {set_laser} (Max possible: {laser_range.max})")
                else: self._logger.warn("  Stereo Module: Cannot set Laser Power.")
            else: self._logger.warn("Stereo Module not found on device.")
            opt_depth_ae = rs.option.enable_auto_exposure; opt_depth_exp = rs.option.exposure; opt_depth_gain = rs.option.gain
            if depth_sensor.supports(opt_depth_ae) and not depth_sensor.is_option_read_only(opt_depth_ae):
                 depth_sensor.set_option(opt_depth_ae, 0); self._logger.info("  Depth Sensor: Auto Exposure Disabled.")
                 if depth_sensor.supports(opt_depth_exp) and not depth_sensor.is_option_read_only(opt_depth_exp): depth_sensor.set_option(opt_depth_exp, 750.0); self._logger.info(f"  Depth Sensor: Manual Exposure set to 750.0")
                 else: self._logger.warn("  Depth Sensor: Cannot set Manual Exposure.")
                 if depth_sensor.supports(opt_depth_gain) and not depth_sensor.is_option_read_only(opt_depth_gain): depth_sensor.set_option(opt_depth_gain, 16.0); self._logger.info(f"  Depth Sensor: Manual Gain set to 16.0")
                 else: self._logger.warn("  Depth Sensor: Cannot set Manual Gain.")
            else: self._logger.warn("  Depth Sensor: Cannot disable Auto Exposure to set manual values.")
        except Exception as e: self._logger.warn(f"Could not set some depth/stereo options: {e}")
        try: # RGB Sensor Manual Exposure/Gain/White Balance
            color_sensor = device.first_color_sensor()
            if color_sensor:
                opt_rgb_ae = rs.option.enable_auto_exposure; opt_rgb_exp = rs.option.exposure; opt_rgb_gain = rs.option.gain; opt_rgb_awb = rs.option.enable_auto_white_balance; opt_rgb_wb = rs.option.white_balance
                if color_sensor.supports(opt_rgb_ae) and not color_sensor.is_option_read_only(opt_rgb_ae):
                    color_sensor.set_option(opt_rgb_ae, 0); self._logger.info("  RGB Sensor: Auto Exposure Disabled.")
                    if color_sensor.supports(opt_rgb_exp) and not color_sensor.is_option_read_only(opt_rgb_exp): color_sensor.set_option(opt_rgb_exp, 500.0); self._logger.info(f"  RGB Sensor: Manual Exposure set to 500.0")
                    else: self._logger.warn("  RGB Sensor: Cannot set Manual Exposure.")
                    if color_sensor.supports(opt_rgb_gain) and not color_sensor.is_option_read_only(opt_rgb_gain): color_sensor.set_option(opt_rgb_gain, 16.0); self._logger.info(f"  RGB Sensor: Manual Gain set to 16.0")
                    else: self._logger.warn("  RGB Sensor: Cannot set Manual Gain.")
                else: self._logger.warn("  RGB Sensor: Cannot disable Auto Exposure to set manual values.")
                if color_sensor.supports(opt_rgb_awb) and not color_sensor.is_option_read_only(opt_rgb_awb):
                     color_sensor.set_option(opt_rgb_awb, 0); self._logger.info("  RGB Sensor: Auto White Balance Disabled.")
                     if color_sensor.supports(opt_rgb_wb) and not color_sensor.is_option_read_only(opt_rgb_wb): color_sensor.set_option(opt_rgb_wb, 5500.0); self._logger.info(f"  RGB Sensor: Manual White Balance set to 5500.0")
                     else: self._logger.warn("  RGB Sensor: Cannot set Manual White Balance.")
                else: self._logger.warn("  RGB Sensor: Cannot disable Auto White Balance to set manual value.")
            else: self._logger.warn("RGB color sensor not found on device.")
        except Exception as e: self._logger.warn(f"Could not set some RGB options: {e}")
    def camera_k(self) -> Optional[np.ndarray]:
        # ... (implementation is the same) ...
        if not self._connected or not self._intrinsics: self._logger.error("Camera not connected or intrinsics unavailable."); return None
        k = self._intrinsics; return np.array([[k.fx, 0, k.ppx], [0, k.fy, k.ppy], [0, 0, 1]], dtype=np.float32)
    def get_intrinsics_object(self) -> Optional[rs.intrinsics]:
        # ... (implementation is the same) ...
        if not self._connected or not self._intrinsics: self._logger.error("Cannot get intrinsics object, camera not connected or intrinsics unavailable."); return None
        return self._intrinsics
    def get_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # ... (implementation is the same) ...
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
        # ... (implementation is the same) ...
        if self._connected: self._logger.info("Disconnecting RealSense camera...");
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
# GraspNet ROS 2 Service Node (Modified)
# =============================================================================
class GraspServiceNode(Node):
    FIRST_ROTATION_OFFSET_QUAT = np.array([ 0.7071068, 0.0, 0.7071068, 0.0 ])
    SECOND_ROTATION_OFFSET_QUAT = np.array([ 0.0, 0.0, 0.7071068, 0.7071068])
    # Use self.MASK_DISTANCE_THRESHOLD instead of local variables
    MASK_DISTANCE_THRESHOLD = 0.017

    def __init__(self, cfgs):
        super().__init__('graspnet_service_node')
        self.cfgs = cfgs # Store config object
        self.callback_group = ReentrantCallbackGroup()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.last_calculated_grasp_matrix = None; self.last_sorted_grasp_group = None
        self.camera = None; self.net = None
        self.last_grasp_obj = None; self.last_cloud = None
        self.visualization_enabled = cfgs.visualize
        self.r_first_offset = None; self.r_second_offset = None
        valid_offsets = True
        try: self.r_first_offset = R.from_quat(self.FIRST_ROTATION_OFFSET_QUAT); self.get_logger().info(f"First fixed rotation offset quat defined: {self.FIRST_ROTATION_OFFSET_QUAT}\nEquivalent First Rotation Matrix:\n{self.r_first_offset.as_matrix()}")
        except ValueError as e: self.get_logger().error(f"Invalid first fixed rotation quat defined: {self.FIRST_ROTATION_OFFSET_QUAT} - {e}"); self.r_first_offset = R.identity(); self.get_logger().warn("Using identity rotation as FIRST offset due to error."); valid_offsets = False
        try: self.r_second_offset = R.from_quat(self.SECOND_ROTATION_OFFSET_QUAT); self.get_logger().info(f"Second fixed rotation offset quat defined: {self.SECOND_ROTATION_OFFSET_QUAT}\nEquivalent Second Rotation Matrix:\n{self.r_second_offset.as_matrix()}")
        except ValueError as e: self.get_logger().error(f"Invalid second fixed rotation quat defined: {self.SECOND_ROTATION_OFFSET_QUAT} - {e}"); self.r_second_offset = R.identity(); self.get_logger().warn("Using identity rotation as SECOND offset due to error."); valid_offsets = False
        if not valid_offsets: self.get_logger().warn("One or both fixed rotation offsets were invalid. TF will reflect only valid offsets.")
        try:
            self.get_logger().info("Initializing Camera Object..."); self.camera = CameraRealsense(serial_number=self.cfgs.serial_number, logger=self.get_logger())
            self.get_logger().info("Initializing GraspNet model..."); self.net = self._get_net()
            self.trigger_service = self.create_service(Trigger,'trigger_grasp_calculation', self.handle_trigger_request, callback_group=self.callback_group); self.get_logger().info(f"Service '{self.trigger_service.srv_name}' ready.")
            if self.cfgs.mask_path: self.get_logger().info(f"Segmentation mask will be loaded from: {self.cfgs.mask_path}")
            else: self.get_logger().info("No segmentation mask path provided, grasp proximity filtering will be skipped.")
            # Access using self. here
            self.get_logger().info(f"Grasp proximity filter threshold: {self.MASK_DISTANCE_THRESHOLD} m")
            self.get_logger().info("GraspNet Service Node initialized and waiting for trigger.")
        except (FileNotFoundError, ImportError, NameError, Exception) as e: self.get_logger().fatal(f"Initialization failed: {e}"); traceback.print_exc(); self._cleanup_and_exit()

    def _cleanup_and_exit(self):
        # ... (implementation is the same) ...
        if rclpy.ok():
            if hasattr(self, '_Node__handle') and self._Node__handle is not None: pass
            else: self.get_logger().warn("Node handle does not exist or is invalid during cleanup.")
            try: context = self.context if hasattr(self, 'context') else None; rclpy.shutdown(context=context); self.get_logger().info("RCLPY shutdown initiated.")
            except Exception as shutdown_e: self.get_logger().error(f"Error during rclpy shutdown: {shutdown_e}")
        sys.exit(1)

    def destroy_node(self):
        # ... (implementation is the same) ...
        self.get_logger().info("Destroying node...")
        if hasattr(self, 'camera') and self.camera: self.camera.disconnect()
        try: super().destroy_node(); self.get_logger().info("Node destruction complete (super call successful).")
        except Exception as e: self.get_logger().error(f"Error during super().destroy_node(): {e}")

    # --- MODIFIED handle_trigger_request ---
    def handle_trigger_request(self, request, response):
        self.get_logger().info("Received trigger request. Starting grasp calculation...")
        grasp_matrix_raw = None; sorted_grasp_group = None; cloud = None
        segmented_points = None; start_time = time.time()

        if self.camera.connected: self.get_logger().warn("Camera was already connected, disconnecting first."); self.camera.disconnect(); time.sleep(0.5)
        connect_success = self.camera.connect(rgb_resolution=(self.cfgs.width, self.cfgs.height), fps=self.cfgs.fps)
        if not connect_success:
            self.get_logger().error("Failed to connect to the camera."); response.success = False; response.message = "Failed to connect to camera."
            if self.camera.connected: self.camera.disconnect()
            return response

        try:
            self.get_logger().info("Calculating grasps (full scene) and loading/processing segmentation mask (if provided)...")
            grasp_matrix_raw, sorted_grasp_group, cloud, segmented_points = self._calculate_best_grasp_and_segment(self.cfgs.mask_path)

            num_grasps_to_show = 0 # Initialize
            if sorted_grasp_group is not None and len(sorted_grasp_group) > 0:
                num_found = len(sorted_grasp_group)
                num_grasps_to_show = min(num_found, 50) # Calculate how many to show
                self.get_logger().info(f"Grasp calculation successful. Found {num_found} valid grasps after all filtering. Will visualize top {num_grasps_to_show}.")
                self.last_calculated_grasp_matrix = grasp_matrix_raw # Still store the best one's matrix for TF
                self.last_sorted_grasp_group = sorted_grasp_group # Store the full filtered group
                self.last_cloud = cloud
                self.last_grasp_obj = sorted_grasp_group[0] # Still store the best Grasp object if needed elsewhere
                response.success = True
                tf_status = "will publish modified pose of BEST grasp." if grasp_matrix_raw is not None else "failed to get BEST grasp matrix for TF."

                if self.cfgs.mask_path:
                    seg_status = f"Found {len(segmented_points)} segmented 3D points from '{os.path.basename(self.cfgs.mask_path)}'." if segmented_points is not None else f"Failed to load/process mask from '{os.path.basename(self.cfgs.mask_path)}'."
                    # Access using self.
                    filt_status = f" Grasps filtered by mask proximity (threshold {self.MASK_DISTANCE_THRESHOLD}m)." if segmented_points is not None else " Mask proximity filter skipped."
                else:
                    seg_status = "Segmentation skipped (no mask path provided)."
                    filt_status = " Mask proximity filter skipped."
                # Updated response message
                response.message = f"Found {num_found} valid grasps. TF {tf_status}{filt_status} {seg_status} Visualization shows top {num_grasps_to_show} grasps."

                if grasp_matrix_raw is not None: self.publish_modified_grasp_tf() # Publish TF for the best grasp
                if segmented_points is not None: self.get_logger().info(f"Successfully obtained {len(segmented_points)} 3D points from segmentation mask.")

                # --- MODIFIED Visualization Block ---
                if self.visualization_enabled and self.last_cloud and self.last_cloud.has_points():
                    self.get_logger().info(f"Attempting blocking visualization (showing top {num_grasps_to_show} grasps)...")
                    try:
                        geoms = [self.last_cloud] # Start with the full scene cloud
                        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]); geoms.append(cam_frame)

                        # Add geometries for the top N grasps
                        if self.last_sorted_grasp_group: # Check if the group exists
                            for i in range(num_grasps_to_show):
                                grasp = self.last_sorted_grasp_group[i]
                                gripper_geom = grasp.to_open3d_geometry()
                                if gripper_geom:
                                    geoms.append(gripper_geom)
                                else:
                                     self.get_logger().warn(f"Could not generate gripper geometry for grasp index {i} (score: {grasp.score:.3f}).")

                        # Add segmented points (if they exist)
                        if segmented_points is not None and len(segmented_points) > 0:
                            seg_pcd = o3d.geometry.PointCloud(); seg_pcd.points = o3d.utility.Vector3dVector(np.array(segmented_points))
                            seg_pcd.paint_uniform_color([1.0, 0.0, 0.0]); geoms.append(seg_pcd)
                            self.get_logger().info(f"Adding {len(segmented_points)} segmented points (red) to visualization.")

                        # Update window title
                        o3d.visualization.draw_geometries(geoms,
                                                            window_name=f"GraspNet Results (Top {num_grasps_to_show} Grasps) & Segmentation",
                                                            width=1280, height=720)
                        self.get_logger().info("Visualization window closed.")
                    except Exception as vis_e: self.get_logger().error(f"Error during visualization: {vis_e}"); traceback.print_exc()
                elif self.visualization_enabled: self.get_logger().warn("Visualization enabled but required cloud missing/empty.")
                # --- END MODIFIED Visualization Block ---

            else: # Handle failure case (no valid grasps found *after all filtering*)
                self.get_logger().warn("Failed to calculate grasp (no valid grasps found after collision and proximity filtering).")
                self.last_calculated_grasp_matrix = None; self.last_sorted_grasp_group = None; self.last_grasp_obj = None; self.last_cloud = cloud # Keep cloud if it exists
                response.success = False; response.message = "Failed to find any valid grasps after filtering."
                if self.cfgs.mask_path: # Add segmentation status even on failure
                     if segmented_points is not None: response.message += f" Found {len(segmented_points)} segmented 3D points."
                     else: response.message += f" Failed to load/process mask from '{os.path.basename(self.cfgs.mask_path)}'."

        except Exception as e:
            self.get_logger().error(f"Exception during grasp calculation pipeline: {e}"); traceback.print_exc()
            self.last_calculated_grasp_matrix = None; self.last_sorted_grasp_group = None; self.last_grasp_obj = None; self.last_cloud = None
            response.success = False; response.message = f"Exception during grasp calculation: {e}"
        finally:
            if self.camera.connected: self.get_logger().info("Disconnecting camera..."); self.camera.disconnect()
            else: self.get_logger().warn("Camera was not connected at the end of handle_trigger_request.")
            end_time = time.time(); self.get_logger().info(f"Grasp trigger request processed in {end_time - start_time:.2f} seconds.")
        return response

    # --- Grasp Calculation Methods ---
    def _get_net(self):
        # ... (implementation is the same) ...
        net = GraspNet(input_feature_dim=0, num_view=self.cfgs.num_view, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); self.get_logger().info(f"Using device: {device}"); net.to(device)
        try: checkpoint = torch.load(self.cfgs.checkpoint_path, map_location=device); net.load_state_dict(checkpoint['model_state_dict']); start_epoch = checkpoint.get('epoch', -1); self.get_logger().info("-> Loaded checkpoint %s (epoch %d)"%(self.cfgs.checkpoint_path, start_epoch))
        except FileNotFoundError as e: self.get_logger().error(f"Checkpoint file not found: {self.cfgs.checkpoint_path}"); raise e
        except KeyError as e: self.get_logger().error(f"Error loading checkpoint: Missing key {e}. Checkpoint might be corrupted or from a different version."); raise e
        except Exception as e: self.get_logger().error(f"Error loading checkpoint: {e}"); traceback.print_exc(); raise e
        net.eval(); return net

    def _get_and_process_data_with_segmentation(self, mask_file_path: Optional[str]):
        # ... (implementation is the same) ...
        self.get_logger().debug("Capture frame, process FULL SCENE for GraspNet, load mask for filtering points...")
        if not self.camera or not self.camera.connected: self.get_logger().error("Camera not connected for data capture."); return None, None, None, None, None, None
        color_bgr, depth_m = self.camera.get_rgbd(); intrinsics = self.camera.get_intrinsics_object()
        if color_bgr is None or depth_m is None or intrinsics is None: self.get_logger().warn("Failed to get RGB-D frame or intrinsics."); return None, None, None, None, None, None
        if color_bgr.size == 0 or depth_m.size == 0: self.get_logger().warn("Received empty color or depth image."); return None, None, None, color_bgr, depth_m, intrinsics
        mask = None; segmented_points_3d = None
        if mask_file_path:
            if not os.path.exists(mask_file_path): self.get_logger().error(f"Segmentation mask file not found at: {mask_file_path}")
            else:
                try:
                    self.get_logger().info(f"Loading segmentation mask from: {mask_file_path}")
                    loaded_mask_img = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
                    if loaded_mask_img is None: self.get_logger().error(f"Failed to load image from {mask_file_path}.")
                    else:
                        self.get_logger().info(f"  Loaded mask image shape: {loaded_mask_img.shape}"); target_shape = depth_m.shape; target_dims = (target_shape[1], target_shape[0])
                        if loaded_mask_img.shape != target_shape: self.get_logger().warn(f"  Resizing loaded mask from {loaded_mask_img.shape} to {target_shape} using INTER_NEAREST."); resized_mask_img = cv2.resize(loaded_mask_img, target_dims, interpolation=cv2.INTER_NEAREST)
                        else: resized_mask_img = loaded_mask_img
                        mask = resized_mask_img > 127; self.get_logger().info(f"  Converted to boolean mask. Found {np.sum(mask)} True pixels.")
                        if mask is not None and np.any(mask): self.get_logger().info("Attempting deprojection using loaded mask..."); segmented_points_3d = deproject_masked_points(mask, depth_m, intrinsics); self.get_logger().info(f"Deprojection complete. Found {len(segmented_points_3d)} 3D points for filtering.")
                        elif mask is not None: self.get_logger().info("Loaded mask is all False, no points for filtering.")
                except Exception as seg_e: self.get_logger().error(f"Exception during mask loading/processing/deprojection step: {seg_e}"); traceback.print_exc(); mask = None; segmented_points_3d = None
        else: self.get_logger().debug("No mask file path provided, skipping segmentation point generation.")
        end_pts = None; collision_cloud = None
        try:
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB); color_norm = color_rgb.astype(np.float32) / 255.0
            intrinsic_matrix = self.camera.camera_k()
            if intrinsic_matrix is None: raise ValueError("Camera intrinsics matrix is None.")
            width = color_bgr.shape[1]; height = color_bgr.shape[0]; k = intrinsic_matrix; fx,fy,cx,cy = k[0,0],k[1,1],k[0,2],k[1,2]; factor_depth=1.0
            self.get_logger().debug("Generating FULL point cloud for GraspNet input & collision...");
            if np.all(depth_m == 0): self.get_logger().warn("Depth image is all zeros."); raise ValueError("Zero depth")
            cam_info = CameraInfo(float(width), float(height), fx, fy, cx, cy, factor_depth)
            full_cloud_organized = create_point_cloud_from_depth_image(depth_m, cam_info, organized=True)
            if full_cloud_organized is None: raise ValueError("Failed to create full point cloud.")
            depth_min = self.cfgs.depth_min; depth_max = self.cfgs.depth_max; valid_depth_mask = (depth_m > depth_min) & (depth_m < depth_max)
            cloud_flat = full_cloud_organized.reshape(-1, 3); color_flat = color_norm.reshape(-1, 3); depth_mask_flat = valid_depth_mask.reshape(-1)
            valid_points_mask = np.isfinite(cloud_flat).all(axis=1); final_mask = depth_mask_flat & valid_points_mask
            cloud_masked = cloud_flat[final_mask]; color_masked = color_flat[final_mask]; num_valid_points = len(cloud_masked)
            self.get_logger().debug(f"Number of valid points in FULL scene after depth/NaN filtering: {num_valid_points}")
            if num_valid_points < self.cfgs.num_point * 0.1: self.get_logger().warn(f"Too few valid points ({num_valid_points}) in full scene! Min required approx: {self.cfgs.num_point * 0.1}.")
            n_sample=self.cfgs.num_point; self.get_logger().debug(f"Sampling {n_sample} points from FULL scene for GraspNet input...");
            if num_valid_points == 0: raise ValueError("Empty full scene cloud after filtering")
            elif num_valid_points >= n_sample: idxs = np.random.choice(num_valid_points, n_sample, replace=False)
            else: self.get_logger().warn(f"Sampling {n_sample} points with replacement from {num_valid_points} available points."); idxs = np.random.choice(num_valid_points, n_sample, replace=True)
            cloud_sampled = cloud_masked[idxs]; end_pts = {}; cloud_torch = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); cloud_torch = cloud_torch.to(dev); end_pts['point_clouds'] = cloud_torch
            collision_cloud = o3d.geometry.PointCloud(); collision_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
            if color_masked.shape[0] == cloud_masked.shape[0]: collision_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
            self.get_logger().debug(f"Full scene data processing successful. Sampled input shape: {cloud_sampled.shape}, Collision cloud points: {len(collision_cloud.points)}")
        except Exception as grasp_proc_e: self.get_logger().error(f"Exception during full scene data processing step: {grasp_proc_e}"); traceback.print_exc(); end_pts = None; collision_cloud = None
        return end_pts, collision_cloud, segmented_points_3d, color_bgr, depth_m, intrinsics

    def _get_grasps(self, end_points):
        # ... (implementation is the same) ...
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
        # ... (implementation is the same - includes fix) ...
        num_grasps_before = len(gg)
        if num_grasps_before == 0: return gg
        if not cloud_o3d or not cloud_o3d.has_points(): self.get_logger().warn("Collision cloud is empty or invalid. Skipping collision detection."); return gg
        self.get_logger().debug(f"Performing collision detection for {num_grasps_before} grasps using full scene cloud...");
        pts=np.asarray(cloud_o3d.points);
        if pts.shape[0]==0: self.get_logger().warn("Collision cloud points array is empty. Skipping collision detection."); return gg
        try:
            detector = ModelFreeCollisionDetector(pts, voxel_size=self.cfgs.voxel_size)
            collision_results = detector.detect(gg, approach_dist=0.005, collision_thresh=self.cfgs.collision_thresh)
            if isinstance(collision_results, (list, tuple)) and len(collision_results) > 0: mask = collision_results[0]
            else: mask = collision_results
            if not isinstance(mask, np.ndarray) or mask.dtype != bool: self.get_logger().error(f"Collision detector did not return a boolean numpy array as expected first element. Got type: {type(mask)}"); raise TypeError("Invalid return type from collision detector")
            n_coll = np.sum(mask); n_valid = num_grasps_before - n_coll; gg_filt = gg[~mask]; self.get_logger().debug(f"Collision check complete: {n_coll} collisions detected, {n_valid} grasps remaining."); return gg_filt
        except IndexError as e: self.get_logger().error(f"IndexError during collision detection: {e}. Check grasp group and points."); traceback.print_exc(); return GraspGroup()
        except (ValueError, TypeError) as e: self.get_logger().error(f"{type(e).__name__} during collision detection (check detector.detect return values/types): {e}"); traceback.print_exc(); return gg
        except Exception as e: self.get_logger().error(f"Unexpected error during collision detection: {e}"); self.get_logger().warn("Returning grasps without collision check due to error."); traceback.print_exc(); return gg


    def _calculate_best_grasp_and_segment(self, mask_file_path: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[GraspGroup], Optional[o3d.geometry.PointCloud], Optional[List[Tuple[float, float, float]]]]:
       # --- Access using self. ---
        t_start = time.time()
        T_graspnet_best_raw = None
        sorted_filtered_gg = None
        full_cloud = None
        segmented_points = None

        try:
            t_d0 = time.time()
            end_pts, full_cloud, segmented_points, _, _, _ = self._get_and_process_data_with_segmentation(mask_file_path)
            t_d1 = time.time()
            self.get_logger().debug(f"Data acquisition & processing: {t_d1 - t_d0:.4f}s")

            if end_pts is None or full_cloud is None:
                self.get_logger().warn("Failed to get valid data/point cloud for GraspNet. Grasp calculation aborted.")
                return None, None, full_cloud, segmented_points

            t_g0 = time.time(); gg = self._get_grasps(end_pts); t_g1 = time.time()
            self.get_logger().debug(f"Grasp inference (full scene): {t_g1 - t_g0:.4f}s. Found {len(gg)} raw grasps.")
            if len(gg) == 0:
                self.get_logger().warn("No grasps predicted by the network.")
                return None, None, full_cloud, segmented_points

            t_c0 = time.time()
            if self.cfgs.collision_thresh > 0: gg_coll_filtered = self._collision_detection(gg, full_cloud)
            else: self.get_logger().info("Collision threshold is 0, skipping collision detection."); gg_coll_filtered = gg
            t_c1 = time.time()
            self.get_logger().debug(f"Collision detection (full scene): {t_c1 - t_c0:.4f}s. {len(gg_coll_filtered)} grasps remain.")
            if len(gg_coll_filtered) == 0:
                self.get_logger().warn("No grasps remaining after collision filtering.")
                return None, None, full_cloud, segmented_points

            gg_final_filtered = None
            if segmented_points and len(segmented_points) > 0:
                 # Access using self.
                self.get_logger().info(f"Filtering {len(gg_coll_filtered)} collision-free grasps by proximity to {len(segmented_points)} segmented points (threshold: {self.MASK_DISTANCE_THRESHOLD}m)...")
                try:
                    segmented_points_np = np.array(segmented_points)
                    if segmented_points_np.shape[0] > 0:
                        kdtree = KDTree(segmented_points_np)
                        filtered_grasps_list = []
                        for grasp in gg_coll_filtered:
                            grasp_center = grasp.translation
                            distance, _ = kdtree.query(grasp_center, k=1)
                             # Access using self.
                            if distance <= self.MASK_DISTANCE_THRESHOLD:
                                filtered_grasps_list.append(grasp.grasp_array)
                        if filtered_grasps_list:
                            filtered_grasps_np = np.vstack(filtered_grasps_list); gg_final_filtered = GraspGroup(filtered_grasps_np)
                            self.get_logger().info(f"Kept {len(gg_final_filtered)} grasps after mask proximity filtering.")
                        else: gg_final_filtered = GraspGroup(); self.get_logger().warn("No grasps remained after mask proximity filtering.")
                    else: self.get_logger().warn("Segmented points array was empty, skipping proximity filtering."); gg_final_filtered = gg_coll_filtered
                except Exception as filter_e: self.get_logger().error(f"Error during mask proximity filtering: {filter_e}"); traceback.print_exc(); gg_final_filtered = gg_coll_filtered
            else: self.get_logger().info("No segmented points available, skipping mask proximity filtering."); gg_final_filtered = gg_coll_filtered

            if len(gg_final_filtered) > 0:
                t_s0 = time.time(); gg_final_filtered.sort_by_score(); sorted_filtered_gg = gg_final_filtered; t_s1 = time.time()
                self.get_logger().debug(f"Final grasp sorting: {t_s1 - t_s0:.4f}s")
            else: self.get_logger().warn("No valid grasps remaining after ALL filtering steps."); sorted_filtered_gg = GraspGroup()

            if len(sorted_filtered_gg) > 0:
                best_grasp_raw = sorted_filtered_gg[0]
                self.get_logger().info(f"Selected BEST grasp (after all filters) -> Score:{best_grasp_raw.score:.4f}, Width:{best_grasp_raw.width:.4f}m, Depth:{best_grasp_raw.depth:.4f}m")
                if hasattr(best_grasp_raw, 'rotation_matrix') and hasattr(best_grasp_raw, 'translation'):
                    T_graspnet_best_raw = np.eye(4); T_graspnet_best_raw[:3, :3] = best_grasp_raw.rotation_matrix; T_graspnet_best_raw[:3, 3] = best_grasp_raw.translation
                    if not np.isfinite(T_graspnet_best_raw).all(): self.get_logger().error("Best grasp resulted in a non-finite RAW transformation matrix! Cannot use for TF."); T_graspnet_best_raw = None
                    else: self.get_logger().debug(f"Best Grasp 4x4 Transform (T_graspnet_best_raw):\n{T_graspnet_best_raw}")
                else: self.get_logger().error("Selected best grasp object is missing 'rotation_matrix' or 'translation' attributes."); T_graspnet_best_raw = None
            else: self.get_logger().warn("Final filtered grasp group is empty, cannot determine best grasp matrix."); T_graspnet_best_raw = None

            t_end = time.time(); grasp_count = len(sorted_filtered_gg) if sorted_filtered_gg is not None else 0
            self.get_logger().info(f"-> Grasp calculation pipeline finished: {t_end - t_start:.4f}s. Found {grasp_count} valid grasps after all filters.")
            return T_graspnet_best_raw, sorted_filtered_gg, full_cloud, segmented_points

        except Exception as e:
            self.get_logger().error(f"Exception in _calculate_best_grasp_and_segment: {e}"); traceback.print_exc()
            return None, None, full_cloud, segmented_points

    def publish_modified_grasp_tf(self):
        # ... (implementation is the same) ...
        if self.last_calculated_grasp_matrix is None or self.r_first_offset is None or self.r_second_offset is None:
            if self.last_calculated_grasp_matrix is not None and (self.r_first_offset is None or self.r_second_offset is None): self.get_logger().warn("Cannot publish TF: One or both fixed rotation offsets are invalid/None.")
            return
        T_graspnet_raw = self.last_calculated_grasp_matrix
        try:
            R_graspnet_matrix_raw = T_graspnet_raw[:3, :3]; t_graspnet_vec_raw = T_graspnet_raw[:3, 3]
            if not np.isfinite(R_graspnet_matrix_raw).all() or not np.isfinite(t_graspnet_vec_raw).all(): self.get_logger().warn("Stored raw grasp matrix has non-finite values. Skipping TF publish."); self.last_calculated_grasp_matrix = None; return
            r_graspnet_raw = R.from_matrix(R_graspnet_matrix_raw); r_modified = r_graspnet_raw * self.r_first_offset * self.r_second_offset; quat_modified = r_modified.as_quat()
            self.get_logger().debug(f"--- Publishing MODIFIED Grasp TF (Chained Offsets) ---"); self.get_logger().debug(f"  Parent Frame: {self.cfgs.camera_link}"); self.get_logger().debug(f"  Child Frame:  estimated_grasp")
            self.get_logger().debug(f"  Raw Translation: [{t_graspnet_vec_raw[0]:.3f}, {t_graspnet_vec_raw[1]:.3f}, {t_graspnet_vec_raw[2]:.3f}] (Published)")
            self.get_logger().debug(f"  FINAL MODIFIED Quat (xyzw): [{quat_modified[0]:.3f}, {quat_modified[1]:.3f}, {quat_modified[2]:.3f}, {quat_modified[3]:.3f}] (Published)")
            self.get_logger().debug(f"------")
            now = self.get_clock().now().to_msg(); t_stamped = TransformStamped(); t_stamped.header.stamp = now; t_stamped.header.frame_id = self.cfgs.camera_link; t_stamped.child_frame_id = 'estimated_grasp'
            t_stamped.transform.translation.x = t_graspnet_vec_raw[0]; t_stamped.transform.translation.y = t_graspnet_vec_raw[1]; t_stamped.transform.translation.z = t_graspnet_vec_raw[2]
            t_stamped.transform.rotation.x = quat_modified[0]; t_stamped.transform.rotation.y = quat_modified[1]; t_stamped.transform.rotation.z = quat_modified[2]; t_stamped.transform.rotation.w = quat_modified[3]
            self.tf_broadcaster.sendTransform(t_stamped); self.get_logger().debug("Modified grasp TF published.")
        except Exception as e: self.get_logger().error(f"Failed during final modified TF publishing: {e}"); traceback.print_exc()

# =============================================================================
# Main Function (Keep as before)
# =============================================================================
def main(args=None):
    # ... (implementation is the same as previous answer, including --mask_path arg) ...
    rclpy.init(args=args)
    parser = argparse.ArgumentParser("GraspNet ROS 2 Service Node")
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--num_point', type=int, default=45000, help="Number of points to sample.")
    parser.add_argument('--num_view', type=int, default=300, help="Number of views used during GraspNet training.")
    parser.add_argument('--collision_thresh', type=float, default=0.01, help="Collision threshold distance [m]. 0 to disable.")
    parser.add_argument('--voxel_size', type=float, default=0.005, help="Voxel size [m] for collision detection.")
    parser.add_argument('--serial_number', type=str, default=None, help="RealSense camera serial number.")
    parser.add_argument('--width', type=int, default=1280, help="Camera image width.")
    parser.add_argument('--height', type=int, default=720, help="Camera image height.")
    parser.add_argument('--fps', type=int, default=15, help="Camera FPS.")
    parser.add_argument('--depth_min', type=float, default=0.1, help="Minimum depth range [m].")
    parser.add_argument('--depth_max', type=float, default=1.5, help="Maximum depth range [m].")
    parser.add_argument('--camera_link', type=str, default='camera_link', help='TF parent frame for the estimated_grasp (optical frame). [default: camera_link]')
    parser.add_argument('--visualize', action='store_true', help="Show Open3D visualization.")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'], help='Set logging level.')
    parser.add_argument('--mask_path', type=str, default=None, help='Optional path to a binary segmentation mask image file. If provided, segmentation points will be calculated.')
    cfgs, _ = parser.parse_known_args()
    log_level_numeric = getattr(logging, cfgs.log_level.upper(), logging.INFO); log_level_ros = getattr(rclpy.logging.LoggingSeverity, cfgs.log_level.upper(), rclpy.logging.LoggingSeverity.INFO)
    rclpy.logging.set_logger_level('graspnet_service_node', log_level_ros); rclpy.logging.set_logger_level('CameraRealsense', log_level_ros)
    logging.basicConfig(level=log_level_numeric, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_logger = rclpy.logging.get_logger("GraspNetNodeMain")
    if not os.path.isfile(cfgs.checkpoint_path): main_logger.fatal(f"Checkpoint file not found: {cfgs.checkpoint_path}"); sys.exit(1)
    else: main_logger.info(f"Using checkpoint: {cfgs.checkpoint_path}")
    if cfgs.mask_path:
        if os.path.isfile(cfgs.mask_path): main_logger.info(f"Using segmentation mask: {cfgs.mask_path}")
        else: main_logger.warn(f"Specified mask path not found: {cfgs.mask_path}. Segmentation will be skipped.")
    else: main_logger.info("No mask path specified, skipping segmentation.")
    # Log the class attribute correctly here
    # main_logger.info(f"Grasp proximity filter threshold: {GraspServiceNode.MASK_DISTANCE_THRESHOLD} m") # Can log the default here
    main_logger.info(f"Target camera frame (parent for grasp): {cfgs.camera_link}")
    main_logger.info(f"Collision check enabled: {cfgs.collision_thresh > 0}, Threshold: {cfgs.collision_thresh}, Voxel Size: {cfgs.voxel_size}")
    main_logger.info(f"Visualization: {'Enabled' if cfgs.visualize else 'Disabled'}")
    main_logger.info(f"Depth Range: [{cfgs.depth_min}, {cfgs.depth_max}] m")
    node = None; executor = SingleThreadedExecutor()
    try:
        node = GraspServiceNode(cfgs); executor.add_node(node)
        # Log the actual threshold being used by the instance
        node.get_logger().info(f"Node using grasp proximity filter threshold: {node.MASK_DISTANCE_THRESHOLD} m")
        node.get_logger().info("GraspNet Service Node spinning... Waiting for trigger on '/trigger_grasp_calculation'. Ctrl+C to stop.")
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
        if rclpy.ok():
            try: rclpy.shutdown(); main_logger.info("RCLPY shutdown complete.")
            except Exception as rclpy_shutdown_e: main_logger.error(f"Error during final RCLPY shutdown: {rclpy_shutdown_e}")
        else: main_logger.info("RCLPY context was already shut down.")
        main_logger.info("Shutdown process finished.")


if __name__ == '__main__':
    main()