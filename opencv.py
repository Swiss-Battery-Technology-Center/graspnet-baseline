      
import cv2
import numpy as np

def load_depth_image_cv2(depth_path):
    """
    Loads a depth image using OpenCV, preserving original bit depth.

    Args:
        depth_path (str): Path to the depth image file (e.g., 16-bit PNG).

    Returns:
        numpy.ndarray: The depth image as a NumPy array, or None if loading fails.
                       The dtype will match the original image (e.g., uint16 for 16-bit PNG).
    """
    # Use cv2.IMREAD_UNCHANGED to load the image as is, including alpha channel
    # and original bit depth (e.g., 16-bit for depth PNGs).
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth_image is None:
        print(f"Error: Could not load depth image from {depth_path}")
        return None

    print(f"Successfully loaded depth image from {depth_path}")
    print(f"Image shape: {depth_image.shape}")
    print(f"Image data type: {depth_image.dtype}")

    return depth_image

# Example usage:
# Replace 'path/to/your/depth_image.png' with the actual path to your file
# For example, using the path from your command:
depth_file_path = '/workspace/ros2/src/graspnet-baseline/doc/example_data/depth_20250430_152951.png'

depth_data = load_depth_image_cv2(depth_file_path)

if depth_data is not None:
    # You can now work with the depth_data numpy array
    # The pixel values represent depth measurements (e.g., in millimeters)
    # You would use the factor_depth from your meta.mat to convert these to meters.

    # Example: Get depth at a specific pixel (row=100, col=200)
    # Check shape first, assuming grayscale depth (H, W)
    if depth_data.ndim == 2:
        pixel_depth_value = depth_data[100, 200]
        print(f"Depth value at pixel (100, 200): {pixel_depth_value}")

        # To convert to meters, you'd need factor_depth from meta.mat
        # For a 16-bit PNG in mm, factor_depth is typically 1000
        pixel_depth_meters = pixel_depth_value / 1
        # print(f"Depth value at pixel (100, 200) in meters (assuming mm data): {pixel_depth_meters}")

    # Note: Displaying 16-bit depth directly with cv2.imshow() is not typical.
    # You usually need to convert it to 8-bit for display (e.g., scale it) or use a color map.
    # For example, scaling for visualization (losing precision):
    max_depth = np.max(depth_data)
    if max_depth > 0:
        depth_display = (depth_data / max_depth * 255).astype(np.uint8)
        cv2.imshow("Scaled Depth Image", depth_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    