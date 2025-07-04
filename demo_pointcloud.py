#!/usr/bin/env python3
""" Demo to show prediction results from a loaded .npy or .npz point cloud.
    Adapted from original GraspNet demo.
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
# import importlib # Not used
# import scipy.io as scio # Not used for .npy/.npz
# from PIL import Image # Not used

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset')) # For GraspNetDataset if it were used directly
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
# from graspnet_dataset import GraspNetDataset # Not used in this modified demo
from collision_detector import ModelFreeCollisionDetector
# from data_utils import CameraInfo, create_point_cloud_from_depth_image # Not used

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--pcd_file_path', required=True, help='Path to the .npy or .npz point cloud file. For .npy, expects (N,3) XYZ or (N,6) XYZRGB. For .npz, expects key "points" with (N,3) or (N,6), and optionally "colors" (N,3) if points are (N,3).')
parser.add_argument('--num_point', type=int, default=85000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.1, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--depth_min', type=float, default=0.3, help='Minimum depth (Z-coordinate) to consider from PCD [default: 0.0, effectively no min filtering if pcd is in sensor frame > 0]')
parser.add_argument('--depth_max', type=float, default=0.56, help='Maximum depth (Z-coordinate) to consider from PCD [default: inf, effectively no max filtering]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path, map_location=device) # Added map_location for CPU fallback
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d) to device %s"%(cfgs.checkpoint_path, start_epoch, device))
    # set model to eval mode
    net.eval()
    return net

def load_and_process_pcd(pcd_file_path):
    print(f"Loading point cloud from: {pcd_file_path}")
    points_xyz = None
    colors_rgb = None # Will store as 0-1 float

    if not os.path.exists(pcd_file_path):
        print(f"Error: File not found {pcd_file_path}")
        return None, None

    try:
        if pcd_file_path.endswith('.npy'):
            data = np.load(pcd_file_path)
            if data.ndim == 2:
                if data.shape[1] == 3: # XYZ
                    points_xyz = data
                    print(f"Loaded {points_xyz.shape[0]} points (XYZ) from .npy file.")
                elif data.shape[1] == 6: # XYZRGB
                    points_xyz = data[:, :3]
                    colors_rgb = data[:, 3:] # Assume 0-1 if from XYZRGB .npy
                    print(f"Loaded {points_xyz.shape[0]} points (XYZRGB) from .npy file.")
                else:
                    print(f"Error: .npy file has unsupported shape {data.shape}. Expected (N,3) or (N,6).")
                    return None, None
            else:
                print(f"Error: .npy file data is not 2D. Shape is {data.shape}.")
                return None, None

        elif pcd_file_path.endswith('.npz'):
            data_dict = np.load(pcd_file_path)
            if 'points' not in data_dict:
                print("Error: .npz file must contain a 'points' array.")
                return None, None
            
            points_data = data_dict['points']
            if points_data.ndim == 2:
                if points_data.shape[1] == 3: # XYZ
                    points_xyz = points_data
                    if 'colors' in data_dict:
                        colors_rgb = data_dict['colors'] # Assume 0-1 if from .npz
                        if colors_rgb.shape[0] != points_xyz.shape[0] or colors_rgb.shape[1] != 3:
                            print(f"Warning: 'colors' array in .npz has mismatched shape {colors_rgb.shape} for points {points_xyz.shape}. Ignoring colors.")
                            colors_rgb = None
                    print(f"Loaded {points_xyz.shape[0]} points (XYZ{' with colors' if colors_rgb is not None else ''}) from .npz file.")
                elif points_data.shape[1] == 6: # XYZRGB
                    points_xyz = points_data[:, :3]
                    colors_rgb = points_data[:, 3:] # Assume 0-1
                    if 'colors' in data_dict:
                        print("Warning: .npz 'points' array is (N,6) XYZRGB, also found 'colors' array. Using colors from 'points'.")
                    print(f"Loaded {points_xyz.shape[0]} points (XYZRGB) from .npz file.")
                else:
                    print(f"Error: .npz 'points' array has unsupported shape {points_data.shape}. Expected (N,3) or (N,6).")
                    return None, None
            else:
                print(f"Error: .npz 'points' array is not 2D. Shape is {points_data.shape}.")
                return None, None
        
        # --- ADDED PLY FILE HANDLING ---
        elif pcd_file_path.endswith('.ply'):
            cloud = o3d.io.read_point_cloud(pcd_file_path)
            if not cloud.has_points():
                print(f"Error: .ply file {pcd_file_path} contains no points.")
                return None, None
            points_xyz = np.asarray(cloud.points)
            if cloud.has_colors():
                colors_rgb = np.asarray(cloud.colors) # Open3D colors are already 0-1 float
                if colors_rgb.shape[0] != points_xyz.shape[0]:
                    print(f"Warning: Mismatch between number of points and colors in .ply file. Ignoring colors.")
                    colors_rgb = None
            print(f"Loaded {points_xyz.shape[0]} points (XYZ{' with colors' if colors_rgb is not None else ''}) from .ply file.")
        # --- END OF PLY FILE HANDLING ---
            
        else:
            # Updated error message to include .ply
            print(f"Error: Unsupported file format {pcd_file_path}. Please use .npy, .npz, or .ply.")
            return None, None
    except Exception as e:
        print(f"Error loading or processing file {pcd_file_path}: {e}")
        return None, None

    if points_xyz is None:
        return None, None

    # Ensure colors are in 0-1 range if they exist (already handled for .ply by Open3D)
    # This block mainly normalizes if colors came from a source that used 0-255.
    if colors_rgb is not None:
        # Check if normalization is needed (e.g., if max value is significantly > 1)
        # This is a heuristic; Open3D .ply colors should already be 0-1.
        # NPY/NPZ might contain 0-255.
        if not pcd_file_path.endswith('.ply') and np.max(colors_rgb) > 1.1: # Added a small tolerance for float comparisons
            print("Normalizing colors from assumed 0-255 range to 0-1.")
            colors_rgb = colors_rgb / 255.0
        colors_rgb = np.clip(colors_rgb, 0.0, 1.0).astype(np.float32)
    else:
        # Create default gray colors if none are provided
        colors_rgb = np.full_like(points_xyz, 0.5, dtype=np.float32) # Use full_like and ensure float32
        print("No color information found or loaded, applying default gray color.")

    # Apply depth filtering (Z-coordinate)
    # ... (rest of depth filtering logic remains the same) ...
    if cfgs.depth_min > 0.0 or cfgs.depth_max < np.inf:
        print(f"Applying depth filter: Z between {cfgs.depth_min} and {cfgs.depth_max}")
        z_coords = points_xyz[:, 2]
        depth_mask = (z_coords >= cfgs.depth_min) & (z_coords <= cfgs.depth_max)
        points_xyz = points_xyz[depth_mask]
        colors_rgb = colors_rgb[depth_mask] # Filter colors accordingly
        if points_xyz.shape[0] == 0:
            print("Error: No points remaining after depth filtering.")
            return None, None
        print(f"{points_xyz.shape[0]} points remaining after depth filtering.")


    # Create Open3D point cloud object (from the full, filtered cloud)
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float32)) # colors_rgb is now always N,3 float32 [0,1]

    # Sample points for GraspNet input
    # ... (rest of sampling logic remains the same) ...
    num_loaded_points = len(points_xyz)
    if num_loaded_points == 0:
        print("Error: No points to sample after loading and filtering.")
        return None, None
        
    if num_loaded_points >= cfgs.num_point:
        idxs = np.random.choice(num_loaded_points, cfgs.num_point, replace=False)
    else:
        print(f"Warning: Number of loaded points ({num_loaded_points}) is less than num_point ({cfgs.num_point}). Sampling with replacement.")
        idxs1 = np.arange(num_loaded_points)
        idxs2 = np.random.choice(num_loaded_points, cfgs.num_point - num_loaded_points, replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    cloud_sampled_xyz = points_xyz[idxs]
    color_sampled_rgb = colors_rgb[idxs] # Sample corresponding colors

    # Prepare end_points for the network
    end_points = dict()
    cloud_sampled_torch = torch.from_numpy(cloud_sampled_xyz[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled_torch = cloud_sampled_torch.to(device)
    end_points['point_clouds'] = cloud_sampled_torch
    end_points['cloud_colors'] = color_sampled_rgb # This is (num_point, 3) RGB float [0,1]

    print(f"Processed data: Sampled {cloud_sampled_xyz.shape[0]} points for GraspNet input. Full cloud has {len(cloud_o3d.points)} points for collision/visualization.")
    return end_points, cloud_o3d


def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points_pred = net(end_points) # net() modifies end_points in-place sometimes
        grasp_preds = pred_decode(end_points_pred)
    
    # Check if grasp_preds is a list/tuple and access the first element
    if isinstance(grasp_preds, (list, tuple)) and len(grasp_preds) > 0:
        gg_array = grasp_preds[0].detach().cpu().numpy()
    elif torch.is_tensor(grasp_preds): # If it's a single tensor (e.g. batch size 1 implicitly handled by pred_decode)
        gg_array = grasp_preds.detach().cpu().numpy()
    else:
        print("Error: Unexpected output format from pred_decode.")
        return GraspGroup() # Return empty GraspGroup

    gg = GraspGroup(gg_array)
    print(f"Predicted {len(gg)} raw grasps.")
    return gg

def collision_detection(gg, scene_cloud_points_np):
    if len(gg) == 0:
        print("No grasps to check for collision.")
        return gg
    if scene_cloud_points_np.shape[0] == 0:
        print("Warning: Scene cloud for collision detection is empty. Skipping collision check.")
        return gg
        
    mfcdetector = ModelFreeCollisionDetector(scene_cloud_points_np, voxel_size=cfgs.voxel_size)
    print(f"Performing collision detection for {len(gg)} grasps with voxel_size {cfgs.voxel_size} and threshold {cfgs.collision_thresh}...")
    # The detect method in ModelFreeCollisionDetector may have specific requirements for approach_dist
    # The original demo uses 0.05. Let's stick to that unless cfgs has an override.
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    
    num_collisions = np.sum(collision_mask)
    gg_filtered = gg[~collision_mask]
    print(f"Collision detection complete: {num_collisions} grasps collided, {len(gg_filtered)} grasps remaining.")
    return gg_filtered

def vis_grasps(gg, cloud_o3d):
    if len(gg) == 0:
        print("No grasps to visualize.")
        if cloud_o3d and cloud_o3d.has_points():
            print("Displaying only the point cloud.")
            o3d.visualization.draw_geometries([cloud_o3d])
        else:
            print("No point cloud to visualize either.")
        return

    gg.nms()
    gg.sort_by_score()
    
    num_to_show = min(len(gg), 100) # Show top 50 after NMS and sorting
    gg_display = gg[:num_to_show]
    
    print(f"Visualizing top {len(gg_display)} grasps (after NMS and sorting).")
    grippers = gg_display.to_open3d_geometry_list()
    
    # Add a coordinate frame for reference if the cloud has points
    geometries_to_draw = []
    if cloud_o3d and cloud_o3d.has_points():
        geometries_to_draw.append(cloud_o3d)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
        geometries_to_draw.append(coord_frame)
    
    geometries_to_draw.extend(grippers)
    o3d.visualization.draw_geometries(geometries_to_draw)

def demo(pcd_file_path_arg):
    net = get_net()
    end_points, cloud_o3d_for_collision_vis = load_and_process_pcd(pcd_file_path_arg)

    if end_points is None or cloud_o3d_for_collision_vis is None:
        print("Failed to load or process point cloud. Exiting demo.")
        return

    gg = get_grasps(net, end_points)
    
    if cfgs.collision_thresh > 0:
        # Pass the points from the Open3D cloud object as a numpy array
        scene_points_for_collision = np.asarray(cloud_o3d_for_collision_vis.points)
        gg = collision_detection(gg, scene_points_for_collision)
    else:
        print("Collision detection skipped as collision_thresh <= 0.")
        
    vis_grasps(gg, cloud_o3d_for_collision_vis)

if __name__=='__main__':
    # Example: python your_script_name.py --checkpoint_path /path/to/checkpoint.tar --pcd_file_path /path/to/your/pointcloud.npy
    if not cfgs.pcd_file_path:
        print("Error: --pcd_file_path argument is required.")
        sys.exit(1)
    if not cfgs.checkpoint_path:
        print("Error: --checkpoint_path argument is required.")
        sys.exit(1)
        
    demo(cfgs.pcd_file_path)