      
import zivid
import numpy as np
import cv2
import time
import logging
import scipy.io as scio
import open3d as o3d
import sys
import os

# The data_utils might be specific to your GraspNet setup.
# If you need create_point_cloud_from_depth_image, ensure this path is correct
# or implement the point cloud creation using open3d directly, which is shown below.
# sys.path.append('/workspace/ros2/src/graspnet-baseline/utils')
# from data_utils import CameraInfo, create_point_cloud_from_depth_image # Assuming these exist

from typing import Tuple

class CameraZivid:
    def __init__(self, serial_number: str = None):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._app = None
        self._camera = None
        self._settings = None # Store settings for capture
        self._intrinsics = None
        self._connected = False
        self._serial_number = serial_number

    def connect(
        self,
        # Zivid doesn't use resolution/FPS like RealSense streams.
        # Capture time depends on settings. Resolution is controlled by ROI or camera mode.
        # We'll use default full resolution or a large ROI.
        # Let's keep parameters as placeholders but use Zivid concepts.
        # Zivid settings are more complex (exposure time, filters, etc.)
        rgb_resolution: Tuple[int, int] = None, # Placeholder, resolution is based on ROI/mode
        depth_resolution: Tuple[int, int] = None, # Placeholder
        fps: int = None # Placeholder, not streaming
    ):
        self._logger.info("Connecting to Zivid camera...")
        try:
            # Create Zivid application
            self._app = zivid.Application()

            # Connect to camera (by serial number if provided)
            if self._serial_number:
                self._camera = self._app.connect_camera(self._serial_number)
                self._logger.info(f"Connected to Zivid camera with serial number: {self._serial_number}")
            else:
                # Connect to the first available camera
                self._camera = self._app.connect_camera()
                self._serial_number = self._camera.info.serial_number # Get serial number of connected camera
                self._logger.info(f"Connected to first available Zivid camera (Serial: {self._serial_number})")

            device_name = self._camera.info.model.value
            self._logger.info(f"Device connected: {device_name}")

            # --- Configure Camera Settings ---
            # Zivid uses Settings objects for configuration before capture
            self._settings = zivid.Settings()

            # Add at least one acquisition setting
            # These control exposure time, gain, brightness, projector power, etc.
            acquisition_settings = zivid.Settings.Acquisition()

            # Example: Configure manual settings similar to RealSense example
            # Note: Zivid settings are different units and scales
            # RealSense exposure 750 (us?) -> Zivid exposure time in microseconds
            # RealSense gain 16 -> Zivid gain (0-32)
            # RealSense white balance 5500 -> Zivid color balance (K)
            # RealSense laser power 360 (max) -> Zivid projector power (0-10)

            # Let's start with auto exposure or specific manual values
            # Recommended: Start with a Zivid YML file or carefully chosen values.
            # For demo, let's try some manual settings:
            acquisition_settings.exposure_time = zivid.ExposureMicroseconds(10000) # Example: 10 ms
            acquisition_settings.gain = 1.0 # Example: Low gain (0.0 to 32.0)
            acquisition_settings.brightness = 1.0 # Example: Moderate brightness (0.0 to 10.0)
            acquisition_settings.auto_exposure_active = False
            acquisition_settings.auto_gain_active = False
            acquisition_settings.aperture = 5.6 # Example aperture

            # Projector (Emitter) settings
            # RealSense max laser power is often ~360mW. Zivid max is 1000mW (power=10).
            # Let's use a moderate power like 8.
            acquisition_settings.projector_power = 8 # Example: 800mW (0-10 scale)
            acquisition_settings.ambient_light_filter = zivid.Settings.Acquisition.AmbientLightFilter.enabled # Often helpful


            self._settings.acquisitions.append(acquisition_settings)

            # --- Configure Processing Settings ---
            # These control filters, color balance, etc.
            self._settings.processing.filters.gaussian.enabled = True
            self._settings.processing.filters.gaussian.sigma = 1.5 # Example sigma

            self._settings.processing.filters.reflection.removal.enabled = True # Remove reflection artifacts
            self._settings.processing.filters.reflection.removal.mode = zivid.Settings.Processing.Filters.Reflection.Removal.Mode.global_

            self._settings.processing.filters.outlier.removal.enabled = True # Remove outliers
            self._settings.processing.filters.outlier.removal.threshold = 5.0 # Example threshold

            # self._settings.processing.filters.cluster.filter.enabled = True # Example: Optional cluster filter
            # self._settings.processing.filters.cluster.filter.min_number_of_points = 3

            # Color settings
            # Zivid usually handles white balance automatically.
            # To match RealSense example, set manual WB:
            self._settings.processing.color.mode = zivid.Settings.Processing.Color.Mode.manual
            # Zivid manual WB expects Kelvin. RealSense 5500K is common for daylight/neutral.
            self._settings.processing.color.balance = zivid.Settings.Processing.Color.Balance(red=1.0, green=1.0, blue=1.0) # You might need to adjust these
            # Note: Zivid color balance isn't a single Kelvin value directly in the API,
            # it's per-channel multipliers. You can try to achieve a desired WB
            # or rely on auto-white balance (`zivid.Settings.Processing.Color.Mode.automatic`).
            # Let's stick to manual mode with neutral balance (1,1,1) or better, use auto.
            self._settings.processing.color.mode = zivid.Settings.Processing.Color.Mode.automatic # Reverting to auto WB is often better

            # --- Optional: Set ROI to match resolution if needed ---
            # Zivid sensors have native resolution (e.g., 1920x1200).
            # You can crop using ROI.
            # If you truly needed 1280x720, you'd define an ROI.
            # Example ROI (center crop if possible, adjust start/end based on sensor):
            # sensor_width = self._camera.info.resolution_max.x # e.g. 1920
            # sensor_height = self._camera.info.resolution_max.y # e.g. 1200
            # target_width, target_height = 1280, 720
            # start_x = (sensor_width - target_width) // 2
            # start_y = (sensor_height - target_height) // 2
            # end_x = start_x + target_width -1
            # end_y = start_y + target_height -1
            # self._settings.roi.manual.enabled = True
            # self._settings.roi.manual.box = zivid.Settings.ROI.Manual.Box(x_min=start_x, y_min=start_y, x_max=end_x, y_max=end_y)
            # Note: This requires precise handling of camera capabilities.
            # For simplicity, let's just use the full sensor or a default large ROI.
            self._settings.roi.manual.enabled = False # Disable manual ROI, use full frame or camera default

            # --- Capture an initial frame to get intrinsics and verify settings ---
            # This applies the settings and captures one frame
            self._logger.info("Capturing initial frame to get intrinsics...")
            try:
                initial_frame = self._camera.capture(self._settings)
                self._intrinsics = initial_frame.camera_intrinsics # Store intrinsics
                self._logger.info("Initial capture successful. Intrinsics obtained.")
            except Exception as e:
                 self._logger.error(f"Failed to capture initial frame: {e}")
                 self._connected = False
                 self._camera.disconnect() # Disconnect on failure
                 self._camera = None
                 self._app = None
                 raise Exception(f"Failed to capture initial frame: {e}")

            self._connected = True
            self._logger.info("Zivid camera connected and configured.")

        except zivid.CameraBySerialNotFound:
            self._logger.error(f"Zivid camera with serial number {self._serial_number} not found.")
            self._connected = False
            raise
        except Exception as e:
            self._logger.error(f"Failed to connect or configure Zivid camera: {e}")
            self._connected = False
            # Attempt to clean up resources even on partial failure
            if self._camera:
                self._camera.disconnect()
                self._camera = None
            if self._app:
                self._app = None
            raise

    def camera_k(self) -> np.ndarray:
        """Return 3x3 intrinsics matrix (for the color camera, inherently aligned)."""
        if not self._connected or self._intrinsics is None:
            raise Exception("Camera not connected or intrinsics not loaded. Call connect() first.")

        # Zivid intrinsics contains the camera matrix K directly
        k_matrix = self._intrinsics.camera_matrix

        return np.array([
            [k_matrix.fx, 0,           k_matrix.cx],
            [0,           k_matrix.fy, k_matrix.cy],
            [0,           0,           1]
        ], dtype=np.float32)

    def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Captures a Zivid frame and extracts color and depth data.

        Returns:
            color_image: HxWx3 (BGR, uint8)
            depth_image: HxW float, in meters
        """
        if not self._connected or self._settings is None:
            raise Exception("Camera not connected or settings not configured. Call connect() first.")

        self._logger.info("Capturing Zivid frame...")
        try:
            # Capture a new frame using the stored settings
            frame = self._camera.capture(self._settings)
            self._logger.info("Zivid frame captured.")

            # Get the point cloud from the frame
            point_cloud = frame.point_cloud()

            # Extract color and depth data
            # copy_colors() returns BGR, uint8
            color_image = point_cloud.copy_colors()

            # copy_depth() returns depth in millimeters (float32 or float64 depending on SDK/camera)
            # We need to convert this to meters as required by the original function signature
            depth_image_mm_float = point_cloud.copy_depth()

            # Convert depth from millimeters to meters
            depth_image_meters = depth_image_mm_float * 0.001 # mm to m

            self._logger.info(f"Captured images - Color shape: {color_image.shape}, Depth shape: {depth_image_meters.shape}")

            return color_image, depth_image_meters

        except Exception as e:
            self._logger.error(f"Failed to capture Zivid frame: {e}")
            return None, None

    def __del__(self):
        """Clean up Zivid resources on object deletion."""
        if self._camera:
            self._logger.info(f"Disconnecting Zivid camera (Serial: {self._serial_number}).")
            self._camera.disconnect()
            self._camera = None
        if self._app:
            # Zivid application should be deleted last
            self._app = None
            self._logger.info("Zivid application released.")
        self._connected = False


####################################
# Main: Capture, Save, Then Re-Load + Show Point Cloud
####################################
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # Use INFO for less verbose Zivid output
    # If you know the serial number, provide it:
    # camera = CameraZivid(serial_number='YOUR_ZIVID_SERIAL_NUMBER')
    # Otherwise, it connects to the first available camera:
    camera = CameraZivid()

    try:
        camera.connect()

        color_img, depth_img_m = camera.get_rgbd()

        if color_img is None or depth_img_m is None:
            logging.warning("Failed to capture images.")
            # No need to call del camera explicitly if using try/finally or context manager
            # `del camera` will happen when the script exits or `camera` goes out of scope
            sys.exit(1) # Exit with error code

        # --- Saving Data ---
        timestamp = int(time.time())
        output_dir = f"capture_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # 1) Save color image (BGR) as PNG
        color_filename = os.path.join(output_dir, "rgb.png")
        cv2.imwrite(color_filename, color_img)
        logging.info(f"Saved color image: {color_filename}")

        # 2) Save depth as 16-bit PNG in millimeters
        # Convert meters (float) to millimeters (uint16) for saving
        # Ensure values are positive and within uint16 range (0 to 65535 mm)
        # Zivid typically provides valid depth, but check range if needed.
        depth_mm_float = depth_img_m * 1000.0
        depth_mm_uint16 = depth_mm_float.astype(np.uint16)
        depth_filename = os.path.join(output_dir, "depth.png")
        cv2.imwrite(depth_filename, depth_mm_uint16)
        logging.info(f"Saved depth image: {depth_filename}")

        # 3) Write meta.mat with intrinsics + factor_depth
        intrinsics = camera.camera_k()  # shape (3,3)
        # factor_depth is how to convert the *saved* depth value (uint16 mm) to meters
        factor_depth_val = 1000       # Saved depth is in mm, so factor is 1000 for meters
        meta_filename = os.path.join(output_dir, "meta.mat")
        scio.savemat(meta_filename, {
            'intrinsic_matrix': intrinsics,
            # Note: factor_depth is usually a single float or a 1x1 array
            'factor_depth': np.array([[factor_depth_val]], dtype=np.float32)
        })
        logging.info(f"Saved {meta_filename} with intrinsics and factor_depth={factor_depth_val}")

        logging.info("Capture and saving complete.")

        # --- Load Saved Data and Show Point Cloud using Open3D ---
        logging.info("Loading saved data and generating point cloud for visualization...")

        # Load images
        loaded_color_img_bgr = cv2.imread(color_filename)
        loaded_depth_img_uint16 = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED) # Read as is (uint16)

        if loaded_color_img_bgr is None or loaded_depth_img_uint16 is None:
             logging.error("Failed to load saved images.")
             sys.exit(1)

        # Load meta data
        meta_data = scio.loadmat(meta_filename)
        loaded_intrinsics = meta_data['intrinsic_matrix']
        loaded_factor_depth = meta_data['factor_depth'][0,0] # Extract scalar

        # Convert depth to meters (float)
        loaded_depth_img_meters_float = loaded_depth_img_uint16.astype(np.float32) / loaded_factor_depth

        # Prepare data for Open3D
        # Convert BGR color image to RGB (Open3D expects RGB) and to Open3D image format
        loaded_color_img_rgb = cv2.cvtColor(loaded_color_img_bgr, cv2.COLOR_BGR2RGB)
        o3d_color_img = o3d.geometry.Image(loaded_color_img_rgb)

        # Convert depth image to Open3D image format (expects float/double)
        o3d_depth_img = o3d.geometry.Image(loaded_depth_img_meters_float)

        # Create Open3D PinholeCameraIntrinsic object
        # fx, fy, cx, cy are columns 0,1,2 and rows 0,1 respectively in the matrix
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=loaded_color_img_rgb.shape[1],
            height=loaded_color_img_rgb.shape[0],
            fx=loaded_intrinsics[0, 0],
            fy=loaded_intrinsics[1, 1],
            cx=loaded_intrinsics[0, 2],
            cy=loaded_intrinsics[1, 2]
        )

        # Create an RGBD image from color and depth
        # depth_scale=1.0 because our loaded_depth_img_meters_float is already in meters
        # depth_trunc=inf means no truncation
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color_img,
            o3d_depth_img,
            depth_scale=1.0,
            depth_trunc=np.inf,
            convert_rgb_to_intensity=False # Keep color
        )

        # Create point cloud from RGBD image and intrinsics
        logging.info("Creating point cloud...")
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d_intrinsic,
            project_valid_depth_only=True # Only project points with valid depth
        )
        logging.info(f"Point cloud created with {len(pcd.points)} points.")

        # Optional: Orient the point cloud if needed (e.g., RealSense often points along +Z)
        # Zivid point cloud is typically oriented with +Z forward.
        # If your application expects a different orientation, apply a transformation.
        # Example: Rotate to align Z-axis upwards (common in some robotics)
        # R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0)) # Rotate 180 deg around X
        # pcd.rotate(R, center=(0, 0, 0))


        # Visualize the point cloud
        logging.info("Visualizing point cloud. Close the visualization window to exit.")
        o3d.visualization.draw_geometries([pcd])

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure camera resources are released
        # The __del__ method handles this when 'camera' object is garbage collected
        # or explicitly deleted, but it's good practice in main blocks to
        # ensure it happens, e.g., with a try...finally block or 'with' statement if implemented.
        pass # `del camera` implicitly happens at end of script or scope

    