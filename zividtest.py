# /workspace/ros2/src/graspnet-baseline/zividtest.py

""" Demo to show prediction results.
    Author: chenxi-wang

    Modified for single-frame input, no workspace_mask, and corrected depth scaling logic.
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
# import importlib # Not used
import scipy.io as scio
from PIL import Image
import time # Added time for logging/timestamps


import torch
from graspnetAPI import GraspGroup

# --- Add project directories to sys.path BEFORE importing local modules ---
# Assuming this script is run from the graspnet-baseline root or similar
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This points to graspnet-baseline

# Assuming required modules are in specific subdirectories relative to graspnet-baseline root
sys.path.append(os.path.join(ROOT_DIR, 'models'))
# sys.path.append(os.path.join(ROOT_DIR, 'dataset')) # dataset module might not be needed if only using data_utils
sys.path.append(os.path.join(ROOT_DIR, 'utils')) # This is where data_utils and collision_detector are expected

# --- Import local modules ---
# These imports must come AFTER the sys.path modifications
try:
    # These should now be imported from the graspnet-baseline repo's structure
    from graspnet import GraspNet, pred_decode
    from collision_detector import ModelFreeCollisionDetector
    from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask # Keep get_workspace_mask import just in case, though not used in demo
    print("Successfully imported local modules for demo.")
except ImportError as e:
    print(f"Error importing local modules. Ensure sys.path is correct ({sys.path}) and modules exist: {e}")
    sys.exit(1)


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
# These point numbers should match what the *model* was trained with.
# 15000 is common for some pipelines, but 20000 was in the original script.
# Using 20000 as default as it was the original demo default.
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]. Number of points to sample for model input.')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]. Number of views for point cloud (parameter for some models).')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')

# Add arguments for specific file paths
parser.add_argument('--rgb_path', required=True, help='Path to the RGB image file')
parser.add_argument('--depth_path', required=True, help='Path to the depth image file (e.g., 16bit mm PNG)')
parser.add_argument('--meta_path', required=True, help='Path to the meta file (.mat with intrinsics and factor_depth)')

cfgs = parser.parse_args()


def get_net():
    """Initializes and loads the GraspNet model."""
    # Init the model
    # Ensure input_feature_dim, num_view, num_angle, num_depth, cylinder_radius, hmin, hmax_list
    # match the model architecture used for training the checkpoint.
    # These values below match the original GraspNet paper/official implementation parameters.
    # Check your checkpoint's training parameters if unsure.
    try:
        net = GraspNet(
            input_feature_dim=0, # Set to 3 if model was trained with RGB colors as features
            num_view=cfgs.num_view,
            num_angle=12, # Usually 12 or 18 angles
            num_depth=4, # Usually 4 depth bins
            cylinder_radius=0.05,
            hmin=-0.02, # Min height below the approach point
            hmax_list=[0.01, 0.02, 0.03, 0.04], # Max height for depth bins
            is_training=False
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
    except Exception as e:
         print(f"Error initializing model: {e}")
         sys.exit(1)


    # Load checkpoint
    print(f"Loading checkpoint from {cfgs.checkpoint_path}...")
    try:
        # Use map_location to ensure checkpoint loads correctly regardless of original save device
        checkpoint = torch.load(cfgs.checkpoint_path, map_location=device)
        # Handle potential mismatch in state dict keys (e.g., 'module.' prefix from DataParallel)
        state_dict = checkpoint['model_state_dict']
        # Example: Remove 'module.' prefix if present
        # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # net.load_state_dict(new_state_dict)
        net.load_state_dict(state_dict) # Try direct load first


        start_epoch = checkpoint.get('epoch', 'N/A') # Get epoch if available
        print("-> loaded checkpoint %s (epoch: %s)"%(cfgs.checkpoint_path, start_epoch))
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {cfgs.checkpoint_path}")
        sys.exit(1)
    except RuntimeError as e:
         print(f"Error loading model state_dict. This could be a mismatch between model architecture and checkpoint weights: {e}")
         sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # set model to eval mode
    net.eval()
    return net

# Modified function to accept file paths directly and handle depth scaling via CameraInfo
def get_and_process_data(rgb_path, depth_path, meta_path):
    """Loads and processes data from specified files."""
    print(f"Loading data from:\nRGB: {rgb_path}\nDepth: {depth_path}\nMeta: {meta_path}")
    # load data
    try:
        color_raw = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0 # Keep raw color [0, 1]
        depth_raw = np.array(Image.open(depth_path)) # Keep raw depth (assuming 16bit mm PNG)
        meta = scio.loadmat(meta_path)
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth'] # This is the meta value (often 1000 for mm->m)

        # Ensure factor_depth is a scalar
        if isinstance(factor_depth, np.ndarray):
             factor_depth = factor_depth.flatten()[0]
        # else: factor_depth is already scalar


    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing data file: {e}")
        sys.exit(1)

    # --- Create CameraInfo object using the factor_depth from meta ---
    # create CameraInfo object
    # CameraInfo takes width, height, fx, fy, cx, cy (all in pixels), and the factor_depth from meta
    # which is often the conversion factor from raw depth unit (e.g. mm) to meters.
    height, width = depth_raw.shape[:2]
    camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    print(f"CameraInfo created with factor_depth: {factor_depth}") # Print to verify

    # --- Pass the raw depth (likely in MM) to create_point_cloud_from_depth_image ---
    # The create_point_cloud_from_depth_image function is expected to use the factor_depth
    # stored *inside* the CameraInfo object to perform the scaling to meters.
    # This call matches the original demo.py call.
    cloud_raw_points = create_point_cloud_from_depth_image(depth_raw, camera, organized=True)


    # get valid points (depth > 0)
    # Use the mask based on the original raw depth PNG where depth > 0 (invalid depth is often 0)
    depth_mask = (depth_raw > 0)

    # --- Removed workspace_mask loading and usage ---
    # mask = (workspace_mask & (depth > 0)) # Original line
    mask = depth_mask # Use only the valid depth mask

    cloud_masked_points = cloud_raw_points[mask] # These points should now be in meters (if CameraInfo factor was correct)
    color_masked = color_raw[mask] # Colors for masked points

    # sample points for model input
    if len(cloud_masked_points) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked_points), cfgs.num_point, replace=False)
    else:
        # Handle case with insufficient valid points
        if len(cloud_masked_points) == 0:
            print("Warning: No valid points found in the depth image after masking.")
            # Return empty data structure compatible with prediction
            # Model input needs [B, N, 3] tensor
            empty_cloud_sampled_tensor = torch.zeros((1, cfgs.num_point, 3), dtype=torch.float32)
            # Return dummy O3D cloud for visualization
            empty_cloud_o3d = o3d.geometry.PointCloud()
            return {'point_clouds': empty_cloud_sampled_tensor.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))}, empty_cloud_o3d

        # Sample with replacement if fewer points than cfgs.num_point
        idxs1 = np.arange(len(cloud_masked_points))
        idxs2 = np.random.choice(len(cloud_masked_points), cfgs.num_point-len(cloud_masked_points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled_points = cloud_masked_points[idxs] # Sampled points in meters, camera frame
    color_sampled = color_masked[idxs] # Sampled colors

    # create Open3D point cloud for visualization and collision detection
    # Use cloud_masked_points which contains ALL valid points (not just sampled) for O3D
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked_points.astype(np.float64)) # Open3D often prefers float64
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float64)) # Open3D expects float [0, 1]

    # convert sampled points to tensor for model input
    # Assuming the model expects a dense [B, N, 3] tensor of points for this demo script's input:
    cloud_sampled_tensor = torch.from_numpy(cloud_sampled_points[np.newaxis].astype(np.float32)) #[1, N, 3]

    # Move tensor to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled_tensor = cloud_sampled_tensor.to(device)

    end_points = dict()
    end_points['point_clouds'] = cloud_sampled_tensor # Input to the model
    # If your model uses colors as features, you might need to add them here,
    # potentially as a separate input key or concatenated to the point cloud.
    # Example: end_points['point_features'] = torch.from_numpy(color_sampled[np.newaxis].astype(np.float32)).to(device)


    return end_points, cloud_o3d # Return sampled points tensor and full O3D cloud


      
def get_grasps(net, end_points):
    """Runs model inference to get grasp predictions."""
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        # pred_decode converts model output to grasp arrays
        # This should output grasp poses in the SAME coordinate frame as the input point cloud (camera frame).
        # The output format is typically [center_x, center_y, center_z, axis_x, axis_y, axis_z, approach_x, approach_y, approach_z, width, depth, score, object_id]
        grasp_preds = pred_decode(end_points) # Assuming this returns a list of grasp arrays (one per batch item)

    # grasp_preds should be a list of numpy arrays or tensors. Get the first item (batch size 1)
    if isinstance(grasp_preds, (list, tuple)):
         if not grasp_preds:
              print("Warning: Model predicted no grasps.")
              gg_array = np.empty((0, 17), dtype=np.float32) # Return empty array structure compatible with GraspGroup
         else:
              # Assuming list contains tensors, detach, move to cpu, convert to numpy
              gg_array = grasp_preds[0].detach().cpu().numpy()
    # Handle case where pred_decode might return a single tensor directly (less common for batches)
    elif isinstance(grasp_preds, torch.Tensor):
         gg_array = grasp_preds.detach().cpu().numpy()
    else:
         print(f"Warning: Unexpected type from pred_decode: {type(grasp_preds)}. Expected list/tuple or tensor.")
         gg_array = np.empty((0, 17), dtype=np.float32)


    # --- PROBABLE FIX FOR GIANT GRASPS (Apply scaling here if needed) ---
    # This entire block is OUTSIDE the if/elif/else above, so it is NOT indented relative to it.

    # Line 248 ->
    if gg_array.shape[0] > 0 and gg_array.shape[1] >= 11: # Check if there are grasps and enough columns
        # The code *inside* this IF must be indented *further* than this line.
        # Since the scaling lines are commented out, Python is seeing an 'if' block with nothing in it.
        # You need to either uncomment and properly indent the scaling lines OR add a 'pass' statement.

        # Assuming width is column 9 and depth is column 10 - VERIFY THIS IN PRED_DECODE
        # If they look like millimeters, uncomment and scale:
        # gg_array[:, 9] = gg_array[:, 9] / 1000.0 # Scale width (assuming mm -> meter)
        # gg_array[:, 10] = gg_array[:, 10] / 1000.0 # Scale depth (assuming mm -> meter)

        pass # <--- ADD A 'pass' STATEMENT HERE IF SCALING LINES REMAIN COMMENTED OUT
             # This provides the required indented block after the 'if'

    # --- THIS LINE MUST BE AT THE SAME INDENTATION LEVEL AS THE 'if', 'elif', 'else' ABOVE ---
    # Line 263 ->
    gg = GraspGroup(gg_array) # Create GraspGroup object from the array (potentially scaled)

    return gg # This line must also be at the same level

    

def collision_detection(gg, cloud_o3d_points_array):
    """Performs collision detection against the point cloud."""
    # ModelFreeCollisionDetector expects a numpy array of points [N, 3] in meters.
    # cloud_o3d_points_array is already the full masked cloud points in meters.
    mfcdetector = ModelFreeCollisionDetector(cloud_o3d_points_array, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask.astype(bool)] # Filter out colliding grasps using boolean mask
    return gg

def vis_grasps(gg, cloud_o3d):
    """Visualizes the point cloud and filtered grasps."""
    gg.nms() # Apply non-maximum suppression (removes redundant grasps)
    gg.sort_by_score() # Sort by confidence score
    gg = gg[:min(50, len(gg))] # Keep top 50 grasps, or fewer if less than 50

    # Convert GraspGroup to list of Open3D gripper geometries (visual representation of grasps)
    # This step uses the width and depth stored IN the GraspGroup (which was created from gg_array)
    try:
        grippers = gg.to_open3d_geometry_list()
    except Exception as e:
         print(f"Error converting grasps to Open3D geometry: {e}")
         grippers = [] # Use empty list if conversion fails


    # Draw the point cloud and grippers
    print(f"Visualizing {len(gg)} grasps on point cloud with {len(cloud_o3d.points)} points...")
    try:
        o3d.visualization.draw_geometries([cloud_o3d, *grippers])
    except Exception as e:
         print(f"Error during Open3D visualization: {e}")


def demo(rgb_path, depth_path, meta_path):
    """Runs the demo pipeline for a single frame."""
    print("\n--- Running Demo ---")
    net = get_net() # Load the model
    end_points, cloud_o3d = get_and_process_data(rgb_path, depth_path, meta_path) # Load and process data

    # Check if data processing failed or resulted in no valid points
    if end_points is None or (isinstance(end_points.get('point_clouds'), torch.Tensor) and end_points['point_clouds'].shape[1] == 0):
        print("Error: Data processing failed or no valid points found. Cannot proceed with grasp prediction.")
        return # Exit if data is not valid

    gg = get_grasps(net, end_points) # Get grasp predictions

    # Collision detection only if threshold > 0 and there are grasps/points to check
    if cfgs.collision_thresh > 0 and len(gg) > 0 and len(cloud_o3d.points) > 0:
        print(f"Running collision detection with threshold: {cfgs.collision_thresh}")
        # Pass the point cloud points as a numpy array
        gg = collision_detection(gg, np.array(cloud_o3d.points))
        print(f"Collision detection finished. {len(gg)} grasps remaining.")

    else:
        print("Collision detection skipped (threshold <= 0, no grasps, or no points).")
        # If collision detection was skipped, potentially sort/NMS here if not done later
        if len(gg) > 0:
            # Note: vis_grasps also performs NMS/sort/top50, so this is slightly redundant
            # but ensures gg is ready if vis_grasps is modified or skipped later.
            gg.nms()
            gg.sort_by_score()
            gg = gg[:min(50, len(gg))] # Keep top 50


    # Ensure there are grasps to visualize
    if len(gg) > 0:
         vis_grasps(gg, cloud_o3d) # Visualize results
    else:
         print("No grasps left after processing/filtering to visualize.")
         print(f"Visualizing only the point cloud with {len(cloud_o3d.points)} points.")
         if len(cloud_o3d.points) > 0:
             o3d.visualization.draw_geometries([cloud_o3d]) # Visualize only cloud if no grasps


    print("\n--- Demo Finished ---")


if __name__=='__main__':
    # Example command (using the paths you provided in the prompt):
    # python zividtest.py --checkpoint_path /workspace/ros2/src/graspnet-baseline/checkpoint-rs.tar \
    #                --rgb_path /workspace/ros2/src/graspnet-baseline/doc/example_data/rgb_20250430_152951.png \
    #                --depth_path /workspace/ros2/src/graspnet-baseline/doc/example_data/depth_20250430_152951.png \
    #                --meta_path /workspace/ros2/src/graspnet-baseline/doc/example_data/meta_20250430_152951.mat \
    #                --num_point 20000 \
    #                --collision_thresh 0.01 \
    #                --voxel_size 0.01

    # Get file paths from command line arguments
    rgb_file = cfgs.rgb_path
    depth_file = cfgs.depth_path
    meta_file = cfgs.meta_path

    # Check if files exist
    if not os.path.exists(rgb_file):
         print(f"Error: RGB file not found at {rgb_file}")
         sys.exit(1)
    if not os.path.exists(depth_file):
         print(f"Error: Depth file not found at {depth_file}")
         sys.exit(1)
    if not os.path.exists(meta_file):
         print(f"Error: Meta file not found at {meta_file}")
         sys.exit(1)

    # Run the demo function with the provided file paths
    demo(rgb_file, depth_file, meta_file)