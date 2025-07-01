import os
import numpy as np
import cv2
import time

# Matplotlib imports for plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class VisualOdometry():
    """
    A class to handle the core Visual Odometry logic, including data loading,
    feature matching, and pose estimation.
    """
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images = self._load_images(os.path.join(data_dir, 'image_l'))
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        Detects features and finds good matches between frame i-1 and i.
        Returns coordinates of good matches and the keypoints from frame i.
        """
        keypoints1, descriptors1 = self.orb.detectAndCompute(self.images[i - 1], None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(self.images[i], None)
        
        matches = []
        if descriptors1 is not None and descriptors2 is not None:
            matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.5 * n.distance:
                    good_matches.append(m)
        except ValueError:
            pass

        q1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        q2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
        
        # Return keypoints from the current frame for visualization
        return q1, q2, keypoints2

    def get_pose(self, q1, q2):
        """Calculates the transformation matrix from feature matches."""
        Essential, mask = cv2.findEssentialMat(q1, q2, self.K)
        R, t = self.decomp_essential_mat(Essential, q1, q2)
        return self._form_transf(R, t)

    def decomp_essential_mat(self, E, q1, q2):
        """Decomposes the Essential matrix to find the correct Rotation and Translation."""
        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1, np.ndarray.flatten(t))
        T2 = self._form_transf(R2, np.ndarray.flatten(t))
        T3 = self._form_transf(R1, np.ndarray.flatten(-t))
        T4 = self._form_transf(R2, np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]

        K = np.concatenate((self.K, np.zeros((3, 1))), axis=1)
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]
        np.set_printoptions(suppress=True)

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)

        max_idx = np.argmax(positives)
        if (max_idx == 2): return R1, np.ndarray.flatten(-t)
        elif (max_idx == 3): return R2, np.ndarray.flatten(-t)
        elif (max_idx == 0): return R1, np.ndarray.flatten(t)
        elif (max_idx == 1): return R2, np.ndarray.flatten(t)

# --- 1. SETUP ---
data_dir = 'KITTI_sequence_2'
vo = VisualOdometry(data_dir)

# Create the figure and subplots. 1 row, 2 columns.
fig, (ax_video, ax_path) = plt.subplots(1, 2, figsize=(15, 6))

# Initialize state variables
current_pose = None
gt_path = []
est_path = []

# Initialize plot elements that will be updated in the animation.
line_gt, = ax_path.plot([], [], 'b-', label='Ground Truth')
line_est, = ax_path.plot([], [], 'g-', label='VO Estimate')
imshow_obj = ax_video.imshow(vo.images[0], cmap='gray')


# --- 2. THE ANIMATION UPDATE FUNCTION ---
def update(frame_index):
    """This function is called by FuncAnimation for each new frame."""
    global current_pose

    # Get the ground truth pose for the current frame
    gt_pose = vo.gt_poses[frame_index]
    
    # Initialize or update the VO pose
    if frame_index == 0:
        current_pose = gt_pose
        current_keypoints = [] # No keypoints to draw on the first frame
    else:
        q1, q2, current_keypoints = vo.get_matches(frame_index)
        if len(q1) > 5:
            try:
                transf = vo.get_pose(q1, q2)
                current_pose = np.matmul(current_pose, np.linalg.inv(transf))
            except (cv2.error, np.linalg.LinAlgError):
                # If pose estimation fails, continue with the previous pose
                pass

    # Append data for path plotting
    gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
    est_path.append((current_pose[0, 3], current_pose[2, 3]))

    # Update path plot data
    gt_x, gt_y = zip(*gt_path)
    est_x, est_y = zip(*est_path)
    line_gt.set_data(gt_x, gt_y)
    line_est.set_data(est_x, est_y)
    ax_path.relim()
    ax_path.autoscale_view()

    # --- ANNOTATE AND UPDATE VIDEO FRAME ---
    # Convert the frame to color to draw colored annotations
    frame_to_display = cv2.cvtColor(vo.images[frame_index], cv2.COLOR_GRAY2BGR)
    # Draw the keypoints found on the current frame in green
    annotated_frame = cv2.drawKeypoints(frame_to_display, current_keypoints, None, color=(0, 255, 0))
    # Update the video frame plot with the annotated image
    imshow_obj.set_data(annotated_frame)

    print(f"Processing frame {frame_index}")

    return [line_gt, line_est, imshow_obj]


# --- 3. SETUP AND RUN THE ANIMATION ---
# Set up plot aesthetics
ax_video.set_title('Video Feed with ORB Features')
ax_video.axis('off')
ax_path.set_title('Path Plot')
ax_path.set_xlabel('x (meters)')
ax_path.set_ylabel('z (meters)')
ax_path.legend()
ax_path.set_aspect('equal', adjustable='box')

# Create the animation object with all our desired settings
ani = animation.FuncAnimation(fig, update, frames=len(vo.images),
                              interval=200,      # Slower speed (200ms delay = 5 FPS)
                              blit=False,        # Needed for changing axis limits
                              repeat=False)      # Run animation only once

# Ensure a tight layout before saving or showing
plt.tight_layout()

# # --- 4. SAVE AND SHOW ---
# # Save the animation as an MP4 video file. Requires FFmpeg.
# try:
#     print("Saving animation to vo_animation.mp4... This may take a moment.")
#     ani.save('vo_animation.mp4', writer='ffmpeg', dpi=150)
#     print("...Animation saved successfully!")
# except Exception as e:
#     print(f"\nCould not save animation. Please ensure FFmpeg is installed and in your system's PATH.")
#     print(f"Error: {e}\n")

# Show the interactive plot window
plt.show()