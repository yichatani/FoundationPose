import pyk4a
from pyk4a import Config, PyK4A
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import cv2

class Camera:
    def __init__(self):
        self.k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                camera_fps=pyk4a.FPS.FPS_30,
                depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
                synchronized_images_only=True,
            )
        )
        self.k4a.start()
        print("Camera initialized and started")
        # Set white balance
        self.k4a.whitebalance = 4500
        assert self.k4a.whitebalance == 4500
        
        print("Warming up camera...")
        for i in range(5):
            capture = self.k4a.get_capture(timeout=2000)
            if capture is not None:
                print(f"Warmup frame {i+1}/5")
        print("Camera ready!")


    def get_k4a_img_depth(self):
        capture = self.k4a.get_capture()
        
        if capture is None:
            raise ValueError("Kinect capture option is None.")
        
        depth_aligned = capture.transformed_depth  # shape: (720, 1280) for 720p
        color_image = capture.color[:, :, :3]  # shape: (720, 1280, 3)
        
        # color_image = color_image[:, 280:1000, :]
        # depth_aligned = depth_aligned[:, 280:1000]
        
        # color_image = cv2.resize(color_image, (256, 256))
        # depth_aligned = cv2.resize(depth_aligned, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        return color_image, depth_aligned
    

    def get_point_cloud(self, capture):
        if capture is not None and capture.color is not None and capture.depth is not None:
            points = capture.depth_point_cloud.reshape((-1, 3))
            colors = capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)) / 255.0 

            # Define bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
            bbox = [-500, -400, -600, 1000, 200, 1500]
            min_bound = np.array(bbox[:3])
            max_bound = np.array(bbox[3:])  

            # Filter points within bbox
            indices = np.all((points >= min_bound) & (points <= max_bound), axis=1)
            points = points[indices]
            colors = colors[indices]
            points = np.concatenate((points, colors), axis=1)

            # Filter points based on the distance threshold
            plane = [0.00, 0.42, 0.91, -537.20]
            distances = self.distance_from_plane(points, plane)
            points = points[distances > 10]

            # downsample
            points = self.downsample_with_fps(points)
            return points
        else:
            raise ValueError("Kinect capture option is None.")
 
    def downsample_with_fps(self, points):
        # fast point cloud sampling using torch3d
        points = torch.from_numpy(points).unsqueeze(0).cuda()
        # points = torch.from_numpy(points).unsqueeze(0)
        self.num_points = torch.tensor([self.num_points]).cuda()
        # self.num_points = torch.tensor([self.num_points])
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=self.num_points)
        points = points.squeeze(0).cpu().numpy()
        points = points[sampled_indices.squeeze(0).cpu().numpy()]
        return points
    
    def k4a_calibration(self):
        # print(dir(self.k4a.calibration))
        color_intrinsics = self.k4a.calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
        depth_intrinsics = self.k4a.calibration.get_camera_matrix(pyk4a.CalibrationType.DEPTH)
        color_distortion = self.k4a.calibration.get_distortion_coefficients(pyk4a.CalibrationType.COLOR)
        depth_distortion = self.k4a.calibration.get_distortion_coefficients(pyk4a.CalibrationType.DEPTH)

        print(f"Color camera intrinsics:\n{color_intrinsics}")
        print(f"Depth camera intrinsics:\n{depth_intrinsics}")
        print(f"Color distortion: {color_distortion}")
        print(f"Depth distortion: {depth_distortion}")


    def visualize_kinect_stream(self, window_name="Kinect Stream"):
        """
        Continune Capture and visualize Kinect's RGB and Depth
        
        Args:
            camera: Camera object
            window_name: 
        """
        print("Starting Kinect visualization...")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        
        frame_count = 0
        
        try:
            while True:
                color_image, depth_aligned = self.get_k4a_img_depth()
                
                # depth vislualize (normalize to 0-255)
                depth_viz = depth_aligned.copy()
                depth_viz = np.where(depth_viz > 0, depth_viz, np.nan)  # set 0 to nan to better vis
                
                # normalize depth to vis
                valid_depth = depth_viz[~np.isnan(depth_viz)]
                if len(valid_depth) > 0:
                    min_depth = np.percentile(valid_depth, 1)
                    max_depth = np.percentile(valid_depth, 99)
                    depth_display = np.clip((depth_viz - min_depth) / (max_depth - min_depth) * 255, 0, 255)
                    depth_display = np.nan_to_num(depth_display, 0).astype(np.uint8)
                else:
                    depth_display = np.zeros_like(depth_viz, dtype=np.uint8)
                
                # add color map to depth
                depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                
                # convert RGB to BGR for OpenCV vis
                color_display = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                
                # horizontal stack RGB and Depth
                combined = np.hstack([color_display, depth_colormap])
                
                # Add Text
                cv2.putText(combined, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, "RGB", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, "Depth (aligned)", (266, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if len(valid_depth) > 0:
                    cv2.putText(combined, f"Depth range: {min_depth:.0f}-{max_depth:.0f}mm", 
                            (266, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # show
                cv2.imshow(window_name, combined)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                # elif key == ord('s'):
                #     cv2.imwrite(f"kinect_rgb_{frame_count:04d}.png", color_display)
                #     cv2.imwrite(f"kinect_depth_{frame_count:04d}.png", depth_colormap)
                #     np.save(f"kinect_depth_{frame_count:04d}.npy", depth_aligned)
                #     print(f"Saved frame {frame_count}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            print(f"Total frames: {frame_count}")


if __name__ == "__main__":

    camera = Camera()
    camera.k4a_calibration()

    # camera.visualize_kinect_stream()