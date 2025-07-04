import numpy as np
import cv2

# Load original npz file
npz_path = "recorded_frames/rgbd_frame.npz"
data = np.load(npz_path)

# Load segmentation PNG (assumed to be grayscale or labeled image)
segmap = cv2.imread("recorded_frames/segmap.png", cv2.IMREAD_UNCHANGED)

# Check size matches depth/rgb
if segmap.shape != data["depth"].shape:
    raise ValueError(f"Segmentation map shape {segmap.shape} does not match depth map shape {data['depth'].shape}")

# Save new npz with segmap added
np.savez("recorded_frames/rgbd_frame_with_seg.npz",
         rgb=data["rgb"],
         depth=data["depth"],
         K=data["K"],
         segmap=segmap)