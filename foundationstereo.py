#!/usr/bin/env python3
""" Demo to show prediction results from a loaded point cloud.
    Can perform segmentation from a 2D PNG mask if camera intrinsics are provided.
    Can also run as a one-shot ROS 2 node to publish the best grasp pose as a TF transform.
"""

import os
import sys
import time
import numpy as np
import open3d as o3d
import argparse
import torch

try:
    import cv2
except ImportError:
    print("ERROR: 'opencv-python' is not installed. Please install it for mask loading:")
    print("pip install opencv-python")
    sys.exit(1)

try:
    from scipy.spatial import KDTree
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("ERROR: 'scipy' is not installed. Please install it:")
    print("pip install scipy")
    sys.exit(1)

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import TransformStamped
    import tf2_ros
except ImportError:
    print("WARNING: 'rclpy' or 'tf2_ros' not found. TF publishing will be unavailable.")
    rclpy = None

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--pcd_file_path', required=True, help='Path to the full scene point cloud file (.npy, .npz, .ply).')
parser.add_argument('--mask_path', type=str, default=None, help='(Optional) Path to a binary PNG segmentation mask image file.')
parser.add_argument('--intrinsics_path', type=str, default=None, help='(Required if --mask_path is used) Path to a .txt file containing camera intrinsics.')
parser.add_argument('--seg_proximity_thresh', type=float, default=0.02, help='Proximity threshold in meters to filter grasps near the segmented object.')
parser.add_argument('--num_point', type=int, default=85000, help='Point Number to sample for GraspNet input.')
parser.add_argument('--num_view', type=int, default=300, help='View Number.')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection.')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size for collision detection.')
parser.add_argument('--depth_min', type=float, default=0.3, help='Minimum depth (Z-coordinate) to consider from PCD.')
parser.add_argument('--depth_max', type=float, default=0.8, help='Maximum depth (Z-coordinate) to consider from PCD.')
parser.add_argument('--visualize', action='store_true', help='Enable visualization of the point cloud and grasps.')
parser.add_argument('--publish_tf', action='store_true', help='Enable publishing the best grasp pose as a TF transform.')
parser.add_argument('--camera_link', type=str, default='camera_color_optical_frame', help='Parent frame ID for the published TF transform.')
cfgs = parser.parse_args()


def publish_grasp_tf(node, broadcaster, transform_matrix, parent_frame):
    # ... (this function is unchanged)
    if not node or not broadcaster:
        return
    t_stamped = TransformStamped()
    t_stamped.header.stamp = node.get_clock().now().to_msg()
    t_stamped.header.frame_id = parent_frame
    t_stamped.child_frame_id = 'estimated_grasp'
    translation = transform_matrix[:3, 3]
    rotation_matrix = transform_matrix[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()
    t_stamped.transform.translation.x = translation[0]
    t_stamped.transform.translation.y = translation[1]
    t_stamped.transform.translation.z = translation[2]
    t_stamped.transform.rotation.x = quat[0]
    t_stamped.transform.rotation.y = quat[1]
    t_stamped.transform.rotation.z = quat[2]
    t_stamped.transform.rotation.w = quat[3]
    broadcaster.sendTransform(t_stamped)

def load_intrinsics_from_file(filepath):
    # ... (this function is unchanged)
    if not os.path.exists(filepath):
        print(f"Error: Intrinsics file not found at {filepath}")
        return None
    try:
        with open(filepath, 'r') as f:
            values = [float(v) for v in f.read().split()]
        if len(values) == 4:
            fx, fy, cx, cy = values
            print(f"Loaded 4 intrinsic values: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            return [fx, fy, cx, cy]
        elif len(values) == 9:
            fx, fy, cx, cy = values[0], values[4], values[2], values[5]
            print(f"Loaded 3x3 intrinsic matrix. Extracted: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            return [fx, fy, cx, cy]
        else:
            print(f"Error: Intrinsics file {filepath} must contain either 4 or 9 numbers. Found {len(values)}.")
            return None
    except (ValueError, IndexError) as e:
        print(f"Error parsing intrinsics file {filepath}: {e}")
        return None

def get_net():
    # ... (this function is unchanged)
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(cfgs.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d) to device %s"%(cfgs.checkpoint_path, start_epoch, device))
    net.eval()
    return net

def load_point_cloud_from_file(pcd_file_path):
    # ... (this function is unchanged)
    if not os.path.exists(pcd_file_path):
        print(f"Error: File not found {pcd_file_path}")
        return None, None
    points_xyz, colors_rgb = None, None
    try:
        if pcd_file_path.endswith('.npy'):
            data = np.load(pcd_file_path)
            if data.ndim != 2 or data.shape[1] not in [3, 6]: raise ValueError(f"Unsupported .npy shape {data.shape}.")
            points_xyz = data[:, :3]
            if data.shape[1] == 6: colors_rgb = data[:, 3:]
        elif pcd_file_path.endswith('.npz'):
            data_dict = np.load(pcd_file_path)
            if 'points' not in data_dict: raise ValueError(".npz file must contain a 'points' array.")
            points_data = data_dict['points']
            if points_data.ndim != 2 or points_data.shape[1] not in [3, 6]: raise ValueError(f"Unsupported .npz 'points' shape {points_data.shape}.")
            points_xyz = points_data[:, :3]
            if points_data.shape[1] == 6: colors_rgb = points_data[:, 3:]
            elif 'colors' in data_dict: colors_rgb = data_dict['colors']
        elif pcd_file_path.endswith('.ply'):
            cloud = o3d.io.read_point_cloud(pcd_file_path)
            if not cloud.has_points(): raise ValueError(".ply file contains no points.")
            points_xyz = np.asarray(cloud.points)
            if cloud.has_colors(): colors_rgb = np.asarray(cloud.colors)
        else:
            raise ValueError(f"Unsupported file format. Please use .npy, .npz, or .ply.")
        print(f"Loaded {points_xyz.shape[0]} points from {os.path.basename(pcd_file_path)}")
        return points_xyz, colors_rgb
    except Exception as e:
        print(f"Error loading or processing file {pcd_file_path}: {e}")
        return None, None

def load_and_process_scene_pcd(pcd_file_path):
    # ... (this function is unchanged)
    points_xyz, colors_rgb = load_point_cloud_from_file(pcd_file_path)
    if points_xyz is None: return None, None
    if colors_rgb is not None and np.max(colors_rgb) > 1.1:
        print("Normalizing colors from assumed 0-255 range to 0-1.")
        colors_rgb /= 255.0
    if colors_rgb is None:
        colors_rgb = np.full_like(points_xyz, 0.5, dtype=np.float32)
        print("No color information found, applying default gray color.")
    print(f"Applying depth filter: Z between {cfgs.depth_min} and {cfgs.depth_max}")
    z_coords = points_xyz[:, 2]
    depth_mask = (z_coords >= cfgs.depth_min) & (z_coords <= cfgs.depth_max)
    points_xyz = points_xyz[depth_mask]
    colors_rgb = colors_rgb[depth_mask]
    if points_xyz.shape[0] == 0:
        print("Error: No points remaining after depth filtering.")
        return None, None
    print(f"{points_xyz.shape[0]} points remaining after depth filtering.")
    scene_cloud_o3d = o3d.geometry.PointCloud()
    scene_cloud_o3d.points = o3d.utility.Vector3dVector(points_xyz)
    scene_cloud_o3d.colors = o3d.utility.Vector3dVector(colors_rgb)
    num_loaded_points = len(points_xyz)
    if num_loaded_points >= cfgs.num_point:
        idxs = np.random.choice(num_loaded_points, cfgs.num_point, replace=False)
    else:
        print(f"Warning: Loaded points ({num_loaded_points}) < num_point ({cfgs.num_point}). Sampling with replacement.")
        idxs = np.random.choice(num_loaded_points, cfgs.num_point, replace=True)
    cloud_sampled_xyz = points_xyz[idxs]
    end_points = dict()
    cloud_sampled_torch = torch.from_numpy(cloud_sampled_xyz[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    end_points['point_clouds'] = cloud_sampled_torch.to(device)
    print(f"Sampled {cloud_sampled_xyz.shape[0]} points for GraspNet input.")
    return end_points, scene_cloud_o3d

def segment_pcd_with_mask(scene_points_np, mask_image, intrinsics):
    # ... (this function is unchanged)
    if scene_points_np is None or scene_points_np.shape[0] == 0:
        print("Error: Scene points are empty for segmentation.")
        return None
    if mask_image is None:
        print("Error: Mask image not loaded for segmentation.")
        return None
    if not intrinsics or len(intrinsics) != 4:
        print("Error: Invalid intrinsics provided for segmentation.")
        return None
    print("Segmenting point cloud using 2D mask and projection...")
    fx, fy, cx, cy = intrinsics
    h, w = mask_image.shape[:2]
    in_front_mask = scene_points_np[:, 2] > 0
    points_in_front = scene_points_np[in_front_mask]
    u = (points_in_front[:, 0] * fx / points_in_front[:, 2]) + cx
    v = (points_in_front[:, 1] * fy / points_in_front[:, 2]) + cy
    bounds_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u_int = u[bounds_mask].astype(int)
    v_int = v[bounds_mask].astype(int)
    pixel_mask = mask_image[v_int, u_int] > 127
    final_points_indices = np.where(in_front_mask)[0][bounds_mask][pixel_mask]
    segmented_points = scene_points_np[final_points_indices]
    print(f"Found {len(segmented_points)} points within the mask.")
    return segmented_points

def get_grasps(net, end_points):
    # ... (this function is unchanged)
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    print(f"Predicted {len(gg)} raw grasps.")
    return gg

def collision_detection(gg, scene_cloud_o3d):
    # ... (this function is unchanged)
    if len(gg) == 0: return gg
    scene_points_np = np.asarray(scene_cloud_o3d.points)
    if scene_points_np.shape[0] == 0:
        print("Warning: Scene cloud for collision is empty. Skipping collision check.")
        return gg
    mfcdetector = ModelFreeCollisionDetector(scene_points_np, voxel_size=cfgs.voxel_size)
    print(f"Performing collision detection for {len(gg)} grasps...")
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg_filtered = gg[~collision_mask]
    print(f"Collision detection complete: {np.sum(collision_mask)} grasps collided, {len(gg_filtered)} grasps remaining.")
    return gg_filtered

def filter_grasps_by_segmentation(gg, seg_points_np, threshold):
    # ... (this function is unchanged)
    if len(gg) == 0: return gg
    if seg_points_np is None or seg_points_np.shape[0] == 0:
        print("No segmented points available, skipping proximity filtering.")
        return gg
    print(f"Filtering {len(gg)} grasps by proximity to {len(seg_points_np)} segmented points (threshold: {threshold}m)...")
    kdtree = KDTree(seg_points_np)
    filtered_grasps_list = []
    for grasp in gg:
        distance, _ = kdtree.query(grasp.translation, k=1)
        if distance <= threshold:
            filtered_grasps_list.append(grasp.grasp_array)
    if not filtered_grasps_list:
        print("Warning: No grasps remained after segmentation proximity filtering.")
        return GraspGroup()
    gg_filtered = GraspGroup(np.vstack(filtered_grasps_list))
    print(f"Kept {len(gg_filtered)} grasps after segmentation proximity filtering.")
    return gg_filtered

def vis_grasps(gg, scene_cloud_o3d, seg_cloud_o3d=None):
    # ... (this function is unchanged)
    if len(gg) == 0:
        print("No grasps to visualize.")
        geometries = []
        if scene_cloud_o3d and scene_cloud_o3d.has_points(): geometries.append(scene_cloud_o3d)
        if seg_cloud_o3d and seg_cloud_o3d.has_points():
            seg_cloud_o3d.paint_uniform_color([1.0, 0, 0])
            geometries.append(seg_cloud_o3d)
        if geometries: o3d.visualization.draw_geometries(geometries, window_name="Scene and Segmentation (No Grasps)")
        else: print("No point cloud to visualize either.")
        return
    gg.nms()
    gg.sort_by_score()
    num_to_show = min(len(gg), 100)
    gg_display = gg[:num_to_show]
    print(f"Visualizing top {len(gg_display)} grasps (after NMS and sorting).")
    grippers = gg_display.to_open3d_geometry_list()
    geometries_to_draw = [scene_cloud_o3d]
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    geometries_to_draw.append(coord_frame)
    if seg_cloud_o3d and seg_cloud_o3d.has_points():
        seg_cloud_o3d.paint_uniform_color([1.0, 0.0, 0.0])
        print("Adding segmented point cloud (in red) to visualization.")
        geometries_to_draw.append(seg_cloud_o3d)
    geometries_to_draw.extend(grippers)
    o3d.visualization.draw_geometries(geometries_to_draw, window_name=f"Grasp Visualization (Top {len(gg_display)})")

def demo(cfgs, node=None):
    # ... (function is mostly unchanged)
    if cfgs.mask_path and not cfgs.intrinsics_path:
        print("Error: --intrinsics_path is required when using --mask_path.")
        sys.exit(1)

    net = get_net()
    
    end_points, scene_cloud_o3d = load_and_process_scene_pcd(cfgs.pcd_file_path)
    if end_points is None:
        print("Failed to load or process scene point cloud. Exiting.")
        return

    seg_points_np = None
    seg_cloud_o3d = None
    if cfgs.mask_path:
        intrinsics = load_intrinsics_from_file(cfgs.intrinsics_path)
        if intrinsics is None:
            print("Failed to load intrinsics. Aborting segmentation.")
            return

        mask_image = cv2.imread(cfgs.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            print(f"Error: Failed to load mask image from {cfgs.mask_path}")
        else:
            scene_points_np = np.asarray(scene_cloud_o3d.points)
            seg_points_np = segment_pcd_with_mask(scene_points_np, mask_image, intrinsics)
            if seg_points_np is not None and len(seg_points_np) > 0:
                seg_cloud_o3d = o3d.geometry.PointCloud()
                seg_cloud_o3d.points = o3d.utility.Vector3dVector(seg_points_np)
            else:
                print("Segmentation resulted in zero points. Grasp filtering will be skipped.")

    gg = get_grasps(net, end_points)
    
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, scene_cloud_o3d)
    else:
        print("Collision detection skipped as collision_thresh <= 0.")

    if seg_points_np is not None:
        gg = filter_grasps_by_segmentation(gg, seg_points_np, cfgs.seg_proximity_thresh)
        
    gg.sort_by_score()

    if cfgs.publish_tf:
        if rclpy is None:
            print("Error: Cannot publish TF because rclpy is not imported.")
        elif node is None:
            print("Error: Cannot publish TF because a ROS 2 node was not provided.")
        elif len(gg) == 0:
            print("No valid grasps found to publish.")
        else:
            best_grasp = gg[0]
            # --- FIX: Manually construct the 4x4 transformation matrix ---
            T_grasp = np.eye(4)
            T_grasp[:3, :3] = best_grasp.rotation_matrix
            T_grasp[:3, 3] = best_grasp.translation
            # --- END FIX ---
            tf_broadcaster = tf2_ros.TransformBroadcaster(node)
            
            print(f"Publishing best grasp pose to TF frame '{cfgs.camera_link}' -> 'estimated_grasp' for 5 seconds...")
            start_time = time.time()
            try:
                while time.time() - start_time < 5.0 and rclpy.ok():
                    publish_grasp_tf(node, tf_broadcaster, T_grasp, cfgs.camera_link)
                    time.sleep(0.1)
                print("Finished publishing TF.")
            except KeyboardInterrupt:
                print("TF publishing interrupted.")

    if cfgs.visualize:
        print("Visualization enabled.")
        vis_grasps(gg, scene_cloud_o3d, seg_cloud_o3d)
    else:
        print("Visualization disabled. Script finished.")

def main():
    # ... (this function is unchanged)
    if cfgs.publish_tf and rclpy is None:
        print("Error: --publish_tf was specified, but rclpy or dependent libraries could not be imported. Please install ROS 2.")
        sys.exit(1)
    node = None
    if cfgs.publish_tf:
        rclpy.init(args=sys.argv)
        node = Node('graspnet_tf_publisher')
        print("ROS 2 node initialized for TF publishing.")
    try:
        demo(cfgs, node=node)
    except Exception as e:
        print(f"An error occurred during the demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            print("Destroying ROS 2 node.")
            node.destroy_node()
        if rclpy is not None and rclpy.ok():
            print("Shutting down ROS 2.")
            rclpy.shutdown()

if __name__=='__main__':
    main()