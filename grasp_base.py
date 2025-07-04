#!/usr/bin/env python3

import numpy as np
import os

pos_base_cam = np.array([0.1483, -0.6155, -0.4126])
quat_base_cam = np.array([0.3981, 0.0859, 0.0224, 0.9130])  # (qx, qy, qz, qw)

def quaternion_to_matrix(q):
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)

def make_4x4(rot3x3, trans3):
    T = np.eye(4, dtype=float)
    T[:3, :3] = rot3x3
    T[:3, 3] = trans3
    return T

def main():
    # 1) Load your "best_grasp.npy" with pickling
    input_file = '/workspace/graspnet-baseline/grasps/best_grasp.npy'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return

    grasp_obj = np.load(input_file, allow_pickle=True)
    best_grasp = grasp_obj.item()  # the actual Grasp object

    # 2) Extract rotation_matrix (3x3) and translation (3,)
    R_cam_grasp = best_grasp.rotation_matrix
    t_cam_grasp = best_grasp.translation
    print("R_cam_grasp:\n", R_cam_grasp)
    print("t_cam_grasp:", t_cam_grasp)

    # 3) Build a 4×4 for camera->grasp
    T_cam_grasp = make_4x4(R_cam_grasp, t_cam_grasp)

    # 4) Build base->camera from your known pos/quat
    R_base_cam = quaternion_to_matrix(quat_base_cam)
    T_base_cam = make_4x4(R_base_cam, pos_base_cam)

    # 5) Multiply to get base->grasp
    T_base_grasp = T_base_cam @ T_cam_grasp

    # 6) Save final numeric 4×4
    output_file = '/workspace/graspnet-baseline/grasps/best_grasp_in_base.npy'
    np.save(output_file, T_base_grasp)
    print(f"Saved best grasp in robot base frame to {output_file}")
    print("Final T_base_grasp:\n", T_base_grasp)

if __name__ == '__main__':
    main()
