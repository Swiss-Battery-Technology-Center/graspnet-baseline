#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import open3d as o3d
import argparse
import torch
import traceback

# --- ROS 2 Imports ---
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.callback_groups import ReentrantCallbackGroup
    from sensor_msgs.msg import PointCloud2, Image, CameraInfo
    from geometry_msgs.msg import TransformStamped
    import tf2_ros
    from cv_bridge import CvBridge
    import message_filters
except ImportError:
    print("ERROR: Could not import ROS 2 libraries."); sys.exit(1)

# --- Other Imports ---
try:
    from scipy.spatial import KDTree
    from scipy.spatial.transform import Rotation as R
    import cv2
except ImportError:
    print("ERROR: scipy or opencv-python is not installed."); sys.exit(1)

# --- GraspNet Imports ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from graspnetAPI import GraspGroup

# =============================================================================
# GraspNet Consumer Node
# =============================================================================
class GraspNetConsumerNode(Node):
    FIRST_ROTATION_OFFSET_QUAT = np.array([ 0.7071068, 0.0, 0.7071068, 0.0 ])
    SECOND_ROTATION_OFFSET_QUAT = np.array([ 0.0, 0.0, 0.7071068, 0.7071068])

    def __init__(self, cfgs):
        super().__init__('graspnet_consumer_node')
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.cfgs = cfgs
        self.get_logger().info("--- GraspNet Consumer Node Initializing ---")
        
        self.net = self._get_net()
        self.cv_bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.last_pcd_time = None
        self.sub_callback_group = ReentrantCallbackGroup()
        
        self.r_first_offset = R.from_quat(self.FIRST_ROTATION_OFFSET_QUAT)
        self.r_second_offset = R.from_quat(self.SECOND_ROTATION_OFFSET_QUAT)
        self.get_logger().info("Initialized fixed rotation offsets for TF publication.")
        
        self.pcd_sub = message_filters.Subscriber(self, PointCloud2, '/perception/points', callback_group=self.sub_callback_group)
        self.mask_sub = message_filters.Subscriber(self, Image, '/perception/mask', callback_group=self.sub_callback_group)
        self.cam_info_sub = message_filters.Subscriber(self, CameraInfo, '/perception/camera_info', callback_group=self.sub_callback_group)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.pcd_sub, self.mask_sub, self.cam_info_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.perception_callback)
        
        self.get_logger().info("Subscribed to points, mask, and camera_info topics.")
        self.get_logger().info(f"TF will be published relative to parent frame: '{self.cfgs.camera_link}'")
        self.get_logger().info(f"Filtering grasps relative to world frame: '{self.cfgs.world_frame}'")
        self.get_logger().info("Waiting for synchronized data to trigger grasp pipeline...")

    def _get_net(self):
        self.get_logger().info("Loading GraspNet model...")
        net = GraspNet(input_feature_dim=0, num_view=self.cfgs.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        checkpoint = torch.load(self.cfgs.checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.get_logger().info(f"-> loaded checkpoint {self.cfgs.checkpoint_path}")
        net.eval()
        return net

    def perception_callback(self, pcd_msg, mask_msg, cam_info_msg):
        self.get_logger().info("Received synchronized PointCloud2, Image, and CameraInfo!")
        current_time = self.get_clock().now()
        if self.last_pcd_time and (current_time - self.last_pcd_time).nanoseconds / 1e9 < 2.0:
            self.get_logger().warn("New data arrived too quickly. Skipping this frame."); return
        self.last_pcd_time = current_time

        try:
            scene_cloud_o3d = self.ros2_pcd_to_o3d(pcd_msg)
            mask_image = self.cv_bridge.imgmsg_to_cv2(mask_msg, "mono8")
            k = cam_info_msg.k
            intrinsics = [k[0], k[4], k[2], k[5]]
            
            end_points, scene_cloud_for_collision = self.process_pcd_for_graspnet(scene_cloud_o3d)
            if end_points is None: return

            seg_points_np = self.segment_pcd_with_mask(np.asarray(scene_cloud_for_collision.points), mask_image, intrinsics)

            gg = self.get_grasps(self.net, end_points)
            if self.cfgs.collision_thresh > 0: gg = self.collision_detection(gg, scene_cloud_for_collision)
            if seg_points_np is not None: gg = self.filter_grasps_by_segmentation(gg, seg_points_np, self.cfgs.seg_proximity_thresh)
            gg.sort_by_score()

            if len(gg) > 0:
                best_grasp_found = False
                for i, grasp in enumerate(gg):
                    # Your excellent world-frame filtering logic starts here
                    r_final = R.from_matrix(grasp.rotation_matrix) * self.r_first_offset * self.r_second_offset
                    final_rotation_matrix = r_final.as_matrix()
                    final_z_cam = final_rotation_matrix[:, 2]
                    
                    try:
                        tf_stamped = self.tf_buffer.lookup_transform(
                            target_frame=self.cfgs.world_frame,
                            source_frame=self.cfgs.camera_link,
                            time=rclpy.time.Time())
                        q = tf_stamped.transform.rotation
                        r_cam2world = R.from_quat([q.x, q.y, q.z, q.w])
                        
                        final_z_world = r_cam2world.apply(final_z_cam)
                        
                        # In a standard Z-up world frame, a negative Z means pointing downwards (good).
                        # A positive Z means pointing upwards (bad).
                        if final_z_world[2] <= self.cfgs.max_world_z_for_approach:
                            self.get_logger().info(f"Accepted grasp #{i} (world Z = {final_z_world[2]:.3f})")
                            
                            # --- THIS IS THE FIX ---
                            # Construct the full 4x4 raw transform matrix to pass to the publisher.
                            T_grasp_raw = np.eye(4)
                            T_grasp_raw[:3,:3] = grasp.rotation_matrix
                            T_grasp_raw[:3,3] = grasp.translation
                            self.publish_grasp_tf(T_grasp_raw, self.cfgs.camera_link)
                            # --- END OF FIX ---
                            
                            best_grasp_found = True
                            break
                        else:
                            self.get_logger().debug(f"Discarding grasp #{i} (world Z = {final_z_world[2]:.3f} > {self.cfgs.max_world_z_for_approach})")
                            continue
                    except tf2_ros.TransformException as ex:
                        self.get_logger().error(f'Could not transform {self.cfgs.camera_link} to {self.cfgs.world_frame}: {ex}')
                        return # Exit callback if transform fails
                
                if not best_grasp_found:
                    self.get_logger().warn("No valid grasps found after applying world orientation filter.")
            else:
                self.get_logger().warn("No valid grasps found after initial pipeline.")

            if self.cfgs.visualize:
                seg_cloud_o3d = None
                if seg_points_np is not None and len(seg_points_np) > 0:
                    seg_cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seg_points_np))
                self.vis_grasps(gg, scene_cloud_for_collision, seg_cloud_o3d)
        except Exception as e:
            tb_str = traceback.format_exc()
            self.get_logger().error(f"Error in perception callback: {e}\n--- Traceback ---\n{tb_str}")

    def ros2_pcd_to_o3d(self, pcd_msg):
        field_names = [field.name for field in pcd_msg.fields]
        cloud_data = np.frombuffer(pcd_msg.data, dtype=np.float32)
        cloud_data = np.reshape(cloud_data, (-1, pcd_msg.point_step // 4))
        xyz = cloud_data[:, [field_names.index('x'), field_names.index('y'), field_names.index('z')]]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        if 'rgb' in field_names:
            rgb_packed = cloud_data[:, field_names.index('rgb')]; rgb_packed.dtype = np.uint32
            r = (rgb_packed >> 16) & 0xFF; g = (rgb_packed >> 8) & 0xFF; b = rgb_packed & 0xFF
            colors = np.dstack((r,g,b)).astype(np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
        return pcd
    
    def process_pcd_for_graspnet(self, scene_cloud_o3d):
        points_xyz = np.asarray(scene_cloud_o3d.points)
        if points_xyz.shape[0] == 0:
            self.get_logger().error("Incoming point cloud has zero points."); return None, None
        z_coords = points_xyz[:, 2]
        depth_mask = (z_coords >= self.cfgs.depth_min) & (z_coords <= self.cfgs.depth_max)
        scene_cloud_o3d = scene_cloud_o3d.select_by_index(np.where(depth_mask)[0])
        points_xyz = np.asarray(scene_cloud_o3d.points) 
        if points_xyz.shape[0] == 0:
            self.get_logger().error("No points remaining after depth filtering."); return None, None
        num_points = len(points_xyz)
        if num_points >= self.cfgs.num_point:
            idxs = np.random.choice(num_points, self.cfgs.num_point, replace=False)
        else:
            self.get_logger().warn(f"Points ({num_points}) < num_point ({self.cfgs.num_point}). Sampling with replacement.")
            idxs = np.random.choice(num_points, self.cfgs.num_point, replace=True)
        cloud_sampled_torch = torch.from_numpy(points_xyz[idxs][np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        end_points = {'point_clouds': cloud_sampled_torch.to(device)}
        return end_points, scene_cloud_o3d

    def publish_grasp_tf(self, raw_transform_matrix, parent_frame):
        """Publishes a TF with a modified rotation for robot compatibility."""
        # This function now correctly receives a 4x4 matrix
        raw_rotation_matrix = raw_transform_matrix[:3, :3]
        raw_translation_vec = raw_transform_matrix[:3, 3]
        r_graspnet_raw = R.from_matrix(raw_rotation_matrix)
        r_modified = r_graspnet_raw * self.r_first_offset * self.r_second_offset
        quat_modified = r_modified.as_quat()

        t_stamped = TransformStamped()
        t_stamped.header.stamp = self.get_clock().now().to_msg()
        t_stamped.header.frame_id = parent_frame
        t_stamped.child_frame_id = 'estimated_grasp'
        
        t_stamped.transform.translation.x = raw_translation_vec[0]
        t_stamped.transform.translation.y = raw_translation_vec[1]
        t_stamped.transform.translation.z = raw_translation_vec[2]
        
        t_stamped.transform.rotation.x = quat_modified[0]
        t_stamped.transform.rotation.y = quat_modified[1]
        t_stamped.transform.rotation.z = quat_modified[2]
        t_stamped.transform.rotation.w = quat_modified[3]
        
        self.tf_broadcaster.sendTransform(t_stamped)

    def segment_pcd_with_mask(self, scene_points_np, mask_image, intrinsics):
        if scene_points_np is None or scene_points_np.shape[0] == 0: return None
        fx, fy, cx, cy = intrinsics; h, w = mask_image.shape[:2]
        in_front_mask = scene_points_np[:, 2] > 0; points_in_front = scene_points_np[in_front_mask]
        u = (points_in_front[:, 0] * fx / points_in_front[:, 2]) + cx
        v = (points_in_front[:, 1] * fy / points_in_front[:, 2]) + cy
        bounds_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u_int, v_int = u[bounds_mask].astype(int), v[bounds_mask].astype(int)
        pixel_mask = mask_image[v_int, u_int] > 127
        final_points_indices = np.where(in_front_mask)[0][bounds_mask][pixel_mask]
        return scene_points_np[final_points_indices]

    def get_grasps(self, net, end_points):
        with torch.no_grad(): end_points = net(end_points); grasp_preds = pred_decode(end_points)
        gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
        return gg

    def collision_detection(self, gg, scene_cloud_o3d):
        if len(gg) == 0: return gg
        scene_points_np = np.asarray(scene_cloud_o3d.points)
        if scene_points_np.shape[0] == 0:
            self.get_logger().warn("Collision cloud is empty. Skipping collision check.")
            return gg
        finite_mask = np.isfinite(scene_points_np).all(axis=1)
        scene_points_finite = scene_points_np[finite_mask]
        if scene_points_finite.shape[0] == 0:
            self.get_logger().warn("Collision cloud has no finite points after cleaning. Skipping.")
            return gg
        mfcdetector = ModelFreeCollisionDetector(scene_points_finite, voxel_size=self.cfgs.voxel_size)
        result = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
        collision_mask = result[0] if isinstance(result, (tuple, list)) else result
        if not (isinstance(collision_mask, np.ndarray) and collision_mask.dtype == bool):
            self.get_logger().error(f"Unexpected collision mask type: {type(collision_mask)}")
            return gg
        n_coll = np.sum(collision_mask)
        self.get_logger().info(f"Collision detection complete. {n_coll} grasps in collision, {len(gg)-n_coll} remaining.")
        return gg[~collision_mask]

    def filter_grasps_by_segmentation(self, gg, seg_points_np, threshold):
        if len(gg) == 0 or seg_points_np is None or seg_points_np.shape[0] == 0: return gg
        kdtree = KDTree(seg_points_np)
        grasp_translations = np.array([g.translation for g in gg])
        distances, _ = kdtree.query(grasp_translations, k=1)
        keep_mask = distances <= threshold
        return GraspGroup(gg.grasp_group_array[keep_mask]) if np.any(keep_mask) else GraspGroup()
    
    def vis_grasps(self, gg, scene_cloud_o3d, seg_cloud_o3d=None):
        if len(gg) == 0:
            geometries = [g for g in [scene_cloud_o3d, seg_cloud_o3d] if g and g.has_points()]
            if geometries: o3d.visualization.draw_geometries(geometries)
            return
        gg.nms(); gg.sort_by_score()
        grippers = gg[:min(len(gg), 100)].to_open3d_geometry_list()
        geometries = [scene_cloud_o3d] + grippers
        if seg_cloud_o3d: seg_cloud_o3d.paint_uniform_color([1,0,0]); geometries.append(seg_cloud_o3d)
        o3d.visualization.draw_geometries(geometries)

# =============================================================================
# Main Function
# =============================================================================
def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--seg_proximity_thresh', type=float, default=0.001)
    parser.add_argument('--num_point', type=int, default=95000)
    parser.add_argument('--num_view', type=int, default=300)
    parser.add_argument('--collision_thresh', type=float, default=0.001)
    parser.add_argument('--voxel_size', type=float, default=0.15)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--depth_min', type=float, default=0.001, help='Minimum depth (Z-coordinate) to consider from PCD.')
    parser.add_argument('--depth_max', type=float, default=1.9, help='Maximum depth (Z-coordinate) to consider from PCD.')
    parser.add_argument('--camera_link', type=str, default='camera_link', help='TF parent frame for the estimated_grasp pose.')
    
    parser.add_argument('--world_frame', type=str, default='world', help='The fixed world frame (with Z-up) for orientation filtering.')
    parser.add_argument(
        '--max_world_z_for_approach', type=float, default=-0.4,
        help="Filters grasps approaching from below. This is the maximum allowed Z-component of the final "
             "grasp's approach vector in the world frame. A value of 0.0 rejects any grasp pointing upwards. "
             "Use a small positive value like 0.1 to allow slightly upward approaches.")
#-0.4 for the pipes
#
    rclpy_args = sys.argv
    cfgs, _ = parser.parse_known_args()

    rclpy.init(args=rclpy_args)
    node = None
    executor = SingleThreadedExecutor()
    try:
        node = GraspNetConsumerNode(cfgs)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        rclpy.logging.get_logger("main").info("Keyboard interrupt received.")
    except Exception as e:
        tb_str = traceback.format_exc()
        rclpy.logging.get_logger("main").fatal(f"Unhandled exception: {e}\n--- Traceback ---\n{tb_str}")
    finally:
        rclpy.logging.get_logger("main").info("Initiating shutdown...")
        if node is not None and rclpy.ok(): node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        rclpy.logging.get_logger("main").info("Shutdown complete.")

if __name__=='__main__':
    main()