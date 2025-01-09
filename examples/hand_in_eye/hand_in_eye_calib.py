# %%
import numpy as np
import os
import sys
import open3d as o3d
import cv2
from tqdm import trange

parent_dir = os.path.dirname(os.getcwd())
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from calibration.hand_in_eye import HandinEyeCalibrator
from calibration.utils import read_data

# %% [markdown]
# # Calibration

# %%
# Read data
base_dir = '../../hand_in_eye2'
rgb_list, depth_list, pose_list, rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs, depth_scale = read_data(base_dir)
print(f"{len(rgb_list)} poses found")
print(f'Camera matrix: {rgb_intrinsics}')

# %%
# Calibrate
charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((5, 5), 0.08, 0.06, charuco_dict)

calibrator = HandinEyeCalibrator(rgb_intrinsics, rgb_coeffs, charuco_dict, board)
R_cam2gripper_avg, t_cam2gripper_avg = calibrator.perform(rgb_list, pose_list)



print("Average Camera to gripper rotation matrix:")
print(R_cam2gripper_avg)
print("Average Camera to gripper translation vector:")
print(t_cam2gripper_avg)

# %% [markdown]
# # Visualize results

# %%
# visualize coordinates
R_avg = R_cam2gripper_avg
t_avg = t_cam2gripper_avg

R_original = np.eye(3)
t_original = np.zeros((3, 1))

R_relative = R_avg.copy()
t_relative = t_avg.copy()

T_original = np.eye(4)
T_original[:3, :3] = R_original
T_original[:3, 3] = t_original.flatten()

T_relative = np.eye(4)
T_relative[:3, :3] = R_relative
T_relative[:3, 3] = t_relative.flatten()

T_transformed = np.dot(T_original, T_relative)

mesh_frame_original = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
mesh_frame_transformed = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)

mesh_frame_original.transform(T_original)
mesh_frame_transformed.transform(T_transformed)

o3d.visualization.draw_geometries([mesh_frame_original, mesh_frame_transformed])

# %%
# accumulate pointclouds
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(rgb_list[0].shape[1], rgb_list[0].shape[0],
                                rgb_intrinsics[0, 0], rgb_intrinsics[1, 1],
                                rgb_intrinsics[0, 2], rgb_intrinsics[1, 2])

# Transformation matrix from camera to gripper
T_cam_to_gripper = np.eye(4)
T_cam_to_gripper[0:3, 0:3] = R_cam2gripper_avg
T_cam_to_gripper[0:3, 3] = t_cam2gripper_avg.flatten()

combined_pcd = o3d.geometry.PointCloud()

# Iterate over all images and add them to the point cloud
for i in range(0,len(rgb_list), 4):
    rgb_img = rgb_list[i]
    depth_img = depth_list[i]
    pose_to_base = pose_list[i]

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_img),
        o3d.geometry.Image(depth_img),
        depth_scale=1 / depth_scale,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intrinsic
    )

    cam_to_world = pose_to_base @ T_cam_to_gripper
    pcd.transform(cam_to_world)

    combined_pcd += pcd

# Visualize the combined point cloud
o3d.visualization.draw_geometries([combined_pcd])

# %% [markdown]
# # TSDF reconstruction

# %%
DEPTH_CUTOFF            = 1
VOXEL_SIZE              =0.005

cam_to_gripper_pose = np.eye(4)
cam_to_gripper_pose[:3, :3] = R_cam2gripper_avg
cam_to_gripper_pose[:3, 3] = t_cam2gripper_avg.squeeze()

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL_SIZE,
    sdf_trunc=3 * VOXEL_SIZE,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

for idx in trange(len(rgb_list)):
    pose = pose_list[idx] @ cam_to_gripper_pose
    rgb = rgb_list[idx]
    rgb = np.ascontiguousarray(rgb)
    depth = depth_list[idx] * depth_scale
    depth[depth > DEPTH_CUTOFF] = 0.0 # remove invalid depth
    depth = np.ascontiguousarray(depth.astype(np.float32))

    rgb = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False)
    intrinsic = camera_intrinsic
    extrinsic = np.linalg.inv(pose)
    # extrinsic = pose
    volume.integrate(rgbd, intrinsic, extrinsic)

# %%
# Get mesh and visualize
mesh = volume.extract_triangle_mesh()
sampled_pcd = mesh.sample_points_uniformly(number_of_points=100000)
o3d.visualization.draw_geometries([sampled_pcd])
o3d.io.write_triangle_mesh("pointcloud.ply", mesh)

# %%



