import cv2
import numpy as np

def visualize_z16_depth(z16_path, max_depth_mm=2000):
    """
    Reads a 16-bit depth PNG in z16 format (where each pixel is in millimeters),
    and visualizes it as a color map for easy inspection.

    :param z16_path: Absolute or relative path to the depth .png file (16-bit).
    :param max_depth_mm: Any depth beyond this is clamped to max_depth_mm for coloring.
    """
    # 1) Read the 16-bit depth image
    depth_raw = cv2.imread(z16_path, -1)  # '-1' means 'load as is' (16-bit)
    if depth_raw is None:
        print(f"Could not read {z16_path}")
        return

    print(f"Loaded depth image shape: {depth_raw.shape}, dtype: {depth_raw.dtype}")

    # 2) Clip to the desired range so near objects are visible
    depth_clipped = np.clip(depth_raw, 0, max_depth_mm)

    # 3) Map [0..max_depth_mm] -> [0..255] as an 8-bit grayscale
    depth_8u = ((depth_clipped / max_depth_mm) * 255).astype(np.uint8)

    # 4) Optionally apply a color map for a heatmap-like visualization
    depth_colored = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

    # 5) Display or save the colorized image
    cv2.imshow("Depth Visualization", depth_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Write out the colorized depth for reference
    out_path = z16_path.replace(".png", "_viz.png")
    cv2.imwrite(out_path, depth_colored)
    print(f"Saved colorized visualization to {out_path}")


if __name__ == "__main__":
    # Hardcode your depth file path and max depth in millimeters here
    depth_file_path = "/workspace/graspnet-baseline/depth_1743003607.png"
    max_depth_in_mm = 800   # e.g. 2 meters

    visualize_z16_depth(depth_file_path, max_depth_in_mm)
