#!/usr/bin/env python3

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
from scipy.spatial.transform import Rotation as R # For rotation matrix to quaternion conversion

import torch
from graspnetAPI import GraspGroup

# --- ROS 2 Imports ---
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros

# --- GraspNet Imports ---
# Ensure these paths are correct relative to where you run the script
# Or install GraspNet code as a Python package
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
# from graspnet_dataset import GraspNetDataset # Not strictly needed for inference demo
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
# --- End GraspNet Imports ---


class GraspPublisher(Node):
    def __init__(self, cfgs, data_dir):
        super().__init__('graspnet_tf_publisher')
        self.cfgs = cfgs
        self.data_dir = data_dir
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.5, self.publish_grasp_tf) # Publish every 0.5 seconds
        self.best_grasp_pose_matrix = None # Store the calculated 4x4 pose matrix

        # --- Initialization (calculate grasp once at startup) ---
        self.get_logger().info("Initializing GraspNet...")
        self.net = self._get_net()
        self.get_logger().info("Processing data and calculating grasp pose...")
        self.best_grasp_pose_matrix = self._calculate_best_grasp()

        if self.best_grasp_pose_matrix is None:
            self.get_logger().error("Failed to calculate initial grasp pose. TF will not be published.")
        else:
            self.get_logger().info("Initial grasp pose calculated successfully. Starting TF broadcast.")
        # --- End Initialization ---

    def _get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0,
                       num_view=self.cfgs.num_view,
                       num_angle=12,
                       num_depth=4,
                       cylinder_radius=0.05,
                       hmin=-0.02,
                       hmax_list=[0.01,0.02,0.03,0.04],
                       is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {device}")
        net.to(device)
        # Load checkpoint
        try:
            checkpoint = torch.load(self.cfgs.checkpoint_path, map_location=device) # Ensure loading to correct device
            net.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            self.get_logger().info("-> Loaded checkpoint %s (epoch: %d)"%(self.cfgs.checkpoint_path, start_epoch))
        except FileNotFoundError:
             self.get_logger().error(f"Checkpoint file not found at {self.cfgs.checkpoint_path}")
             raise
        except Exception as e:
             self.get_logger().error(f"Error loading checkpoint: {e}")
             raise
        # set model to eval mode
        net.eval()
        return net

    def _get_and_process_data(self):
        # load data
        try:
            color = np.array(Image.open(os.path.join(self.data_dir, 'rgb_1743178613.png')), dtype=np.float32) / 255.0
            depth = np.array(Image.open(os.path.join(self.data_dir, 'depth_1743178613.png')))
            workspace_mask = np.array(Image.open(os.path.join(self.data_dir, 'workspace_mask.png')))
            meta = scio.loadmat(os.path.join(self.data_dir, 'meta_1743178613.mat'))
        except FileNotFoundError as e:
            self.get_logger().error(f"Error loading data file: {e}. Check data_dir: {self.data_dir}")
            raise
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        # generate cloud
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        mask = (workspace_mask > 0) & (depth > 0) # Ensure mask is boolean
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # sample points
        if len(cloud_masked) == 0:
            self.get_logger().error("No valid points found in the masked point cloud!")
            raise ValueError("No valid points after masking.")

        if len(cloud_masked) >= self.cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), self.cfgs.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.cfgs.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        # color_sampled = color_masked[idxs] # Colors not needed for GraspNet input, but maybe for collision cloud

        # convert data for network
        end_points = dict()
        cloud_sampled_torch = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled_torch = cloud_sampled_torch.to(device)
        end_points['point_clouds'] = cloud_sampled_torch
        # end_points['cloud_colors'] = color_sampled # Not used by the network

        # Prepare Open3D cloud for collision detection
        collision_cloud = o3d.geometry.PointCloud()
        # Use cloud_masked for collision check for better accuracy than sampled
        collision_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        # collision_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32)) # Optional

        return end_points, collision_cloud

    def _get_grasps(self, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        if grasp_preds[0] is None or len(grasp_preds[0]) == 0:
             self.get_logger().warn("Network did not predict any grasps.")
             return GraspGroup() # Return empty GraspGroup
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def _collision_detection(self, gg, cloud):
        if len(gg) == 0:
            return gg # No grasps to check
        if len(cloud.points) == 0:
            self.get_logger().warn("Collision cloud is empty. Skipping collision detection.")
            return gg

        mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud.points), voxel_size=self.cfgs.voxel_size)
        try:
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
            num_collisions = np.sum(collision_mask)
            self.get_logger().info(f"Collision detection: {num_collisions}/{len(gg)} grasps removed.")
            gg = gg[~collision_mask]
        except Exception as e:
            self.get_logger().error(f"Error during collision detection: {e}")
            # Decide how to handle: return all grasps, no grasps, or raise?
            # Returning all grasps might be safer if collision detection fails unexpectedly.
            self.get_logger().warn("Collision detection failed, returning grasps without collision check.")
            # Or return empty if collision checking is critical:
            # return GraspGroup()
        return gg

    def _calculate_best_grasp(self):
        """Calculates the best grasp pose and returns it as a 4x4 numpy matrix."""
        try:
            end_points, collision_cloud = self._get_and_process_data()
            gg = self._get_grasps(end_points)
            self.get_logger().info(f"Generated {len(gg)} grasps initially.")

            if self.cfgs.collision_thresh > 0:
                gg = self._collision_detection(gg, collision_cloud)
                self.get_logger().info(f"{len(gg)} grasps remaining after collision detection.")

            if len(gg) == 0:
                self.get_logger().warn("No valid grasps found after filtering.")
                return None

            # Sort by score and pick the best one
            gg.sort_by_score()
            best_grasp = gg[0]
            self.get_logger().info(f"Selected best grasp with score: {best_grasp.score:.4f}")

            # --- Create the 4x4 homogeneous transformation matrix ---
            T = np.eye(4)
            # Ensure attributes exist (they should for GraspGroup elements)
            if hasattr(best_grasp, 'rotation_matrix') and hasattr(best_grasp, 'translation'):
                T[:3, :3] = best_grasp.rotation_matrix
                T[:3, 3] = best_grasp.translation
                self.get_logger().info(f"Best grasp pose matrix:\n{T}")
                return T
            else:
                self.get_logger().error("Selected grasp object missing 'rotation_matrix' or 'translation'.")
                return None

        except Exception as e:
            self.get_logger().error(f"Error during grasp calculation pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

    def publish_grasp_tf(self):
        """Publishes the stored best grasp pose as a TF transform."""
        if self.best_grasp_pose_matrix is None:
            # self.get_logger().warn("No grasp pose available to publish.", throttle_duration_sec=5) # Prevent spamming logs
            return

        T = self.best_grasp_pose_matrix
        now = self.get_clock().now().to_msg()

        t_stamped = TransformStamped()
        t_stamped.header.stamp = now
        # IMPORTANT: This frame_id should be the frame the point cloud is relative to.
        # Often 'camera_depth_optical_frame' for clouds from RGBD sensors.
        t_stamped.header.frame_id = 'camera_depth_optical_frame'
        t_stamped.child_frame_id = 'estimated_grasp' # Name for the grasp pose frame

        # Translation
        t_stamped.transform.translation.x = T[0, 3]
        t_stamped.transform.translation.y = T[1, 3]
        t_stamped.transform.translation.z = T[2, 3]

        # Rotation (Matrix to Quaternion using scipy)
        try:
            r = R.from_matrix(T[:3, :3])
            quat = r.as_quat() # Returns (x, y, z, w)
            t_stamped.transform.rotation.x = quat[0]
            t_stamped.transform.rotation.y = quat[1]
            t_stamped.transform.rotation.z = quat[2]
            t_stamped.transform.rotation.w = quat[3]
        except Exception as e:
             self.get_logger().error(f"Failed to convert rotation matrix to quaternion: {e}")
             return # Don't publish if rotation is invalid

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t_stamped)
        # self.get_logger().info(f"Published grasp TF: {t_stamped.child_frame_id} -> {t_stamped.header.frame_id}", throttle_duration_sec=5)

def main(args=None):
    rclpy.init(args=args)

    # --- Argument Parsing ---
    # We need to parse arguments before creating the node
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold [default: 0.01]')
    parser.add_argument('--voxel_size', type=float, default=0.001, help='Voxel Size for collision detection [default: 0.001]')
    parser.add_argument('--data_dir', type=str, default='doc/example_data', help='Directory containing input data')
    # Use parse_known_args to ignore ROS-specific arguments like --ros-args
    cfgs, _ = parser.parse_known_args()
    # --- End Argument Parsing ---

    # Validate data directory existence early
    if not os.path.isdir(cfgs.data_dir):
        print(f"[ERROR] Data directory not found: {cfgs.data_dir}", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)
     # Validate checkpoint existence early
    if not os.path.isfile(cfgs.checkpoint_path):
        print(f"[ERROR] Checkpoint file not found: {cfgs.checkpoint_path}", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)

    grasp_publisher_node = None
    try:
        grasp_publisher_node = GraspPublisher(cfgs, cfgs.data_dir)
        rclpy.spin(grasp_publisher_node)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, shutting down.")
    except Exception as e:
         # Log any exceptions during node creation or spinning
         if grasp_publisher_node:
             grasp_publisher_node.get_logger().fatal(f"Unhandled exception: {e}")
         else:
             print(f"[FATAL] Unhandled exception before node creation: {e}", file=sys.stderr)
         import traceback
         traceback.print_exc()
    finally:
        # Cleanup
        if grasp_publisher_node:
            grasp_publisher_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()