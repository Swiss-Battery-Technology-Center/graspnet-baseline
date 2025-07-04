#!/usr/bin/env python3

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import time
import logging
import traceback
from PIL import Image # For loading image files
import scipy.io as scio # For loading .mat files
from typing import Tuple, Optional, List 

# --- PyTorch Import ---
import torch

# --- GraspNet Imports ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add paths for GraspNet model and utils
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset')) # For CameraInfo, etc.
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

try:
    from graspnet import GraspNet, pred_decode
    # Handle GraspNetAPI path robustly
    grasp_api_path_candidates = [
        os.path.join(ROOT_DIR, 'utils', 'graspnetAPI'),
        os.path.join(ROOT_DIR, 'graspnetAPI'), # If API is a sibling to 'utils'/'models'
        os.path.join(ROOT_DIR, 'utils') # If GraspNetAPI contents are directly in utils
    ]
    api_path_found = False
    for path_candidate in grasp_api_path_candidates:
        # Check for a common file within graspnetAPI to confirm its presence
        if os.path.exists(os.path.join(path_candidate, 'grasp.py')) or \
           os.path.exists(os.path.join(path_candidate, 'graspnetAPI', '__init__.py')):
            if path_candidate not in sys.path:
                 sys.path.append(path_candidate)
            api_path_found = True
            print(f"Found and added GraspNetAPI at: {path_candidate}")
            break
    if not api_path_found and os.path.join(ROOT_DIR) not in sys.path: # Last resort if API is at ROOT_DIR
        sys.path.append(ROOT_DIR)
        print(f"Added ROOT_DIR to sys.path for GraspNetAPI: {ROOT_DIR}")


    from graspnetAPI import GraspGroup
    from collision_detector import ModelFreeCollisionDetector
    from data_utils import CameraInfo, create_point_cloud_from_depth_image
except ImportError as e:
    print(f"Import Error: {e}\nSys Path: {sys.path}\nEnsure GraspNet and its submodules (like graspnetAPI) are correctly installed and accessible.")
    sys.exit(1)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraspPipeline:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.net = self._get_net()
        self.best_grasp_matrix = None
        self.all_grasps = None
        self.scene_cloud = None

    def _get_net(self):
        logger.info("Initializing GraspNet model...")
        net = GraspNet(input_feature_dim=0, num_view=self.cfgs.num_view, num_angle=12, num_depth=4,
                        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        net.to(device)
        try:
            checkpoint = torch.load(self.cfgs.checkpoint_path, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"-> Loaded checkpoint {self.cfgs.checkpoint_path} (epoch {checkpoint.get('epoch', -1)})")
        except FileNotFoundError: logger.error(f"Checkpoint file not found: {self.cfgs.checkpoint_path}"); raise
        except KeyError as e: logger.error(f"Bad checkpoint: Missing key {e}"); raise
        except Exception as e: logger.error(f"Error loading checkpoint: {e}"); traceback.print_exc(); raise
        net.eval()
        return net

    def _load_and_process_data(self, data_dir: str) -> Tuple[Optional[dict], Optional[o3d.geometry.PointCloud]]:
        logger.info(f"Loading and processing data from directory: {data_dir}")
        try:
            color_path = os.path.join(data_dir, self.cfgs.color_img_name)
            depth_path = os.path.join(data_dir, self.cfgs.depth_img_name)
            meta_path = os.path.join(data_dir, self.cfgs.meta_file_name)
            
            if not os.path.isfile(color_path): raise FileNotFoundError(f"Color image not found: {color_path}")
            if not os.path.isfile(depth_path): raise FileNotFoundError(f"Depth image not found: {depth_path}")
            if not os.path.isfile(meta_path): raise FileNotFoundError(f"Meta file not found: {meta_path}")

            color_img_pil = Image.open(color_path)
            color_img = np.array(color_img_pil, dtype=np.float32) / 255.0
            depth_img = np.array(Image.open(depth_path))
            
            meta = scio.loadmat(meta_path)
            if 'intrinsic_matrix' not in meta: raise ValueError("Meta file missing 'intrinsic_matrix'")
            if 'factor_depth' not in meta: raise ValueError("Meta file missing 'factor_depth'")
            
            intrinsic_matrix = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth'].item()

            if color_img.shape[2] == 4: # RGBA to RGB
                color_img = color_img[:, :, :3]

            height, width = depth_img.shape[:2]
            fx, fy = intrinsic_matrix[0,0], intrinsic_matrix[1,1]
            cx, cy = intrinsic_matrix[0,2], intrinsic_matrix[1,2]
            
            camera_info = CameraInfo(width, height, fx, fy, cx, cy, factor_depth)
            cloud_organized_xyz = create_point_cloud_from_depth_image(depth_img, camera_info, organized=True)

            if cloud_organized_xyz is None: logger.error("Failed to create point cloud."); return None, None

            mask = (depth_img > 0)
            workspace_mask_path = os.path.join(data_dir, self.cfgs.workspace_mask_name)
            if self.cfgs.use_workspace_mask and os.path.isfile(workspace_mask_path):
                logger.info(f"Using workspace mask: {workspace_mask_path}")
                workspace_mask_img = np.array(Image.open(workspace_mask_path).convert('L'))
                mask = workspace_mask_img & mask
            elif self.cfgs.use_workspace_mask:
                logger.warn(f"Workspace mask enabled but not found: {workspace_mask_path}")

            cloud_masked_xyz = cloud_organized_xyz[mask]
            color_masked_rgb = color_img[mask]

            depth_values_m = cloud_masked_xyz[:, 2]
            depth_range_mask = (depth_values_m >= self.cfgs.depth_min) & (depth_values_m <= self.cfgs.depth_max)
            
            cloud_filtered_xyz = cloud_masked_xyz[depth_range_mask]
            color_filtered_rgb = color_masked_rgb[depth_range_mask]

            num_valid_points = len(cloud_filtered_xyz)
            if num_valid_points == 0: logger.error("No valid points after filtering."); return None, None
            logger.info(f"{num_valid_points} points remaining after filtering.")

            n_sample = self.cfgs.num_point
            if num_valid_points >= n_sample: idxs = np.random.choice(num_valid_points, n_sample, replace=False)
            else:
                logger.warn(f"Sampling {n_sample} with replacement from {num_valid_points} points.")
                idxs = np.random.choice(num_valid_points, n_sample, replace=True)
            
            cloud_sampled_for_gn = cloud_filtered_xyz[idxs]
            end_pts = {'point_clouds': torch.from_numpy(cloud_sampled_for_gn[np.newaxis].astype(np.float32)).to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))}
            
            processed_cloud_o3d = o3d.geometry.PointCloud()
            processed_cloud_o3d.points = o3d.utility.Vector3dVector(cloud_filtered_xyz.astype(np.float32))
            processed_cloud_o3d.colors = o3d.utility.Vector3dVector(color_filtered_rgb.astype(np.float32))
            
            logger.debug(f"Processed data. Sampled cloud: {cloud_sampled_for_gn.shape}. Full cloud: {len(processed_cloud_o3d.points)} points.")
            return end_pts, processed_cloud_o3d

        except FileNotFoundError as e: logger.error(f"File not found: {e}"); raise
        except ValueError as e: logger.error(f"Value error: {e}"); raise
        except Exception as e: logger.error(f"Error in _load_and_process_data: {e}"); traceback.print_exc(); return None, None

    def _get_grasps(self, end_points):
        logger.debug("Inferring grasps...")
        if not end_points or 'point_clouds' not in end_points or end_points['point_clouds'].nelement() == 0:
            logger.error("Invalid end_points for inference."); return GraspGroup()
        try:
            with torch.no_grad(): pred_out = self.net(end_points); grasp_preds = pred_decode(pred_out)
            if not grasp_preds or grasp_preds[0] is None: logger.warn("GraspNet prediction None/empty."); return GraspGroup()
            gg_array = grasp_preds[0].detach().cpu().numpy()
            if gg_array.size == 0: logger.warn("GraspNet predicted empty grasp array."); return GraspGroup()
            gg = GraspGroup(gg_array)
            logger.debug(f"Network predicted {len(gg)} raw grasps."); return gg
        except Exception as e: logger.error(f"Grasp inference/decoding error: {e}"); traceback.print_exc(); return GraspGroup()

    def _collision_detection(self, gg, cloud_o3d):
        if len(gg) == 0: return gg
        if not cloud_o3d or not cloud_o3d.has_points(): logger.warn("Empty collision cloud."); return gg
        logger.debug(f"Collision detection for {len(gg)} grasps...")
        pts = np.asarray(cloud_o3d.points)
        if pts.shape[0] == 0: logger.warn("Collision cloud points array empty."); return gg
        try:
            detector = ModelFreeCollisionDetector(pts, voxel_size=self.cfgs.voxel_size)
            mask, _ = detector.detect(gg, approach_dist=0.01, collision_thresh=self.cfgs.collision_thresh)
            gg_filt = gg[~mask]
            logger.debug(f"Collision check: {np.sum(mask)} collisions, {len(gg_filt)} grasps remaining."); return gg_filt
        except Exception as e: logger.error(f"Collision detection error: {e}"); traceback.print_exc(); return gg

    def run(self):
        logger.info(f"Starting grasp calculation for directory: {self.cfgs.input_dir}")
        start_time = time.time()
        
        end_pts, self.scene_cloud = self._load_and_process_data(self.cfgs.input_dir)
        if end_pts is None or self.scene_cloud is None:
            logger.error("Failed to load or process data. Aborting grasp calculation.")
            return False

        gg = self._get_grasps(end_pts)
        if len(gg) == 0:
            logger.warn("No grasps predicted by the network.")
            self.all_grasps = GraspGroup() # Ensure it's an empty group
            return False

        if self.cfgs.collision_thresh > 0:
            self.all_grasps = self._collision_detection(gg, self.scene_cloud)
        else:
            logger.info("Collision detection skipped (threshold <= 0).")
            self.all_grasps = gg
        
        if len(self.all_grasps) == 0:
            logger.warn("No grasps remaining after collision filtering (if any).")
            return False

        self.all_grasps.sort_by_score() # Sorts in place
        best_grasp_obj = self.all_grasps[0]
        
        logger.info(f"Found {len(self.all_grasps)} valid grasps. Best grasp score: {best_grasp_obj.score:.4f}, width: {best_grasp_obj.width:.4f}m")
        
        # Store the 4x4 matrix of the best grasp
        if hasattr(best_grasp_obj, 'rotation_matrix') and hasattr(best_grasp_obj, 'translation'):
            self.best_grasp_matrix = np.eye(4)
            self.best_grasp_matrix[:3, :3] = best_grasp_obj.rotation_matrix
            self.best_grasp_matrix[:3, 3] = best_grasp_obj.translation
            if not np.isfinite(self.best_grasp_matrix).all():
                logger.error("Best grasp resulted in a non-finite transformation matrix!")
                self.best_grasp_matrix = None
            else:
                logger.info(f"Best Grasp Transformation Matrix:\n{self.best_grasp_matrix}")
        else:
            logger.error("Best grasp object missing 'rotation_matrix' or 'translation'.")
            self.best_grasp_matrix = None
            
        end_time = time.time()
        logger.info(f"Grasp pipeline finished in {end_time - start_time:.2f} seconds.")
        return True

    def visualize_results(self):
        if not self.cfgs.visualize:
            logger.info("Visualization disabled by config.")
            return
        if self.scene_cloud is None or not self.scene_cloud.has_points():
            logger.warn("No scene cloud to visualize.")
            return
        if self.all_grasps is None or len(self.all_grasps) == 0:
            logger.warn("No grasps to visualize. Displaying only point cloud.")
            o3d.visualization.draw_geometries([self.scene_cloud])
            return

        logger.info("Attempting visualization (showing top 50 valid grasps)...")
        try:
            geoms = [self.scene_cloud]
            # Add a coordinate frame representing the camera/PCD origin
            sensor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
            geoms.append(sensor_frame)

            # NMS and sort again just before visualization if desired (already sorted in run)
            # self.all_grasps.nms()
            # self.all_grasps.sort_by_score()
            
            num_grasps_to_show = min(len(self.all_grasps), 50) # Show top 50
            logger.info(f"Visualizing the top {num_grasps_to_show} grasps.")

            for i in range(num_grasps_to_show):
                grasp = self.all_grasps[i]
                gripper_geom = grasp.to_open3d_geometry()
                if gripper_geom:
                    geoms.append(gripper_geom)
            
            o3d.visualization.draw_geometries(geoms,
                                                window_name=f"GraspNet Results (Top {num_grasps_to_show} Grasps)",
                                                width=1280, height=720)
            logger.info("Visualization window closed.")
        except Exception as vis_e:
            logger.error(f"Error during visualization: {vis_e}")
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser("GraspNet from Directory Input (Non-ROS)")
    # Model and GraspNet processing args
    parser.add_argument('--checkpoint_path', required=True, help='GraspNet model checkpoint path.')
    parser.add_argument('--num_point', type=int, default=20000, help="Number of points to sample for GraspNet input.")
    parser.add_argument('--num_view', type=int, default=300, help="Number of views (GraspNet hyperparameter).")
    parser.add_argument('--collision_thresh', type=float, default=0.01, help="Collision threshold [m]. <=0 disables.")
    parser.add_argument('--voxel_size', type=float, default=0.008, help="Voxel size [m] for collision detection.")
    
    # File input related args
    parser.add_argument('--input_dir', required=True, help="Directory containing color.png, depth.png, meta.mat.")
    parser.add_argument('--color_img_name', type=str, default='color.png', help="Name of the color image file.")
    parser.add_argument('--depth_img_name', type=str, default='depth.png', help="Name of the depth image file.")
    parser.add_argument('--meta_file_name', type=str, default='meta.mat', help="Name of the .mat metadata file.")
    parser.add_argument('--workspace_mask_name', type=str, default='workspace_mask.png', help="Name of the optional workspace mask file.")
    parser.add_argument('--use_workspace_mask', action='store_true', help="Enable usage of workspace mask if available.")
    parser.add_argument('--depth_min', type=float, default=0.2, help="Minimum depth range [m] to consider from files.")
    parser.add_argument('--depth_max', type=float, default=1.2, help="Maximum depth range [m] to consider from files.")

    # Visualization args
    parser.add_argument('--visualize', action='store_true', help="Show Open3D visualization of point cloud and grasps.")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG','INFO','WARN','ERROR','FATAL'])
    
    cfgs = parser.parse_args()

    # Configure logging level for the script's logger
    logger.setLevel(getattr(logging, cfgs.log_level.upper()))

    if not os.path.isfile(cfgs.checkpoint_path): logger.fatal(f"Checkpoint missing: {cfgs.checkpoint_path}"); sys.exit(1)
    if not os.path.isdir(cfgs.input_dir): logger.fatal(f"Input directory missing: {cfgs.input_dir}"); sys.exit(1)
    
    logger.info(f"Using Checkpoint: {cfgs.checkpoint_path}")
    logger.info(f"Input File Directory: {cfgs.input_dir}")
    logger.info(f"Visualization: {'Enabled' if cfgs.visualize else 'Disabled'}")

    try:
        pipeline = GraspPipeline(cfgs)
        success = pipeline.run()
        if success:
            logger.info("Grasp calculation successful.")
            # You can access pipeline.best_grasp_matrix and pipeline.all_grasps here
            if cfgs.visualize:
                pipeline.visualize_results()
        else:
            logger.error("Grasp calculation failed.")
            if cfgs.visualize and pipeline.scene_cloud: # Still try to show cloud if calc failed
                logger.info("Displaying scene cloud (if available) despite grasp failure.")
                o3d.visualization.draw_geometries([pipeline.scene_cloud])


    except FileNotFoundError as fnf_e: logger.fatal(f"Initialization or processing error (FileNotFound): {fnf_e}")
    except Exception as e: logger.fatal(f"Unhandled exception in main execution: {e}"); traceback.print_exc()

if __name__ == '__main__':
    main()