# %% [markdown]
# # Connect

# %%
# Panda hostname/IP and Desk login information of your robot
hostname = '169.254.37.13'
username = 'admin'
password = 'admin1234'

# panda-py is chatty, activate information log level
import logging
logging.basicConfig(level=logging.INFO)

# %%
import panda_py

desk = panda_py.Desk(hostname, username, password, platform='fr3')
# desk.unlock()
# desk.activate_fci()

# %%
from panda_py import libfranka

panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)

# %% [markdown]
# # Hand in eye calib

# %%
from panda_py import constants
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# %%
def generate_and_move_to_pose(init_pose, roll, pitch, yaw, z_add, x_add, y_add, max_roll_deviation, max_pitch_deviation, max_yaw_deviation):
    """Generate a new pose with turbulence and move the robot arm to it."""
    roll_turbulent = roll + np.random.uniform(-max_roll_deviation, max_roll_deviation)
    pitch_turbulent = pitch + np.random.uniform(-max_pitch_deviation, max_pitch_deviation)
    yaw_turbulent = yaw + np.random.uniform(-max_yaw_deviation, max_yaw_deviation)

    r = R.from_euler('xyz', [roll_turbulent, pitch_turbulent, yaw_turbulent], degrees=False)
    rotation_matrix = r.as_matrix()

    absolute_rotation_matrix = np.dot(init_pose[:3, :3], rotation_matrix)

    pose = init_pose.copy()
    pose[:3, :3] = absolute_rotation_matrix
    pose[2, 3] += z_add
    pose[0, 3] += x_add
    pose[1, 3] += y_add

    panda.move_to_pose(pose)
    
    return pose

def save_pose(pose, base_dir, frame_num):
    """Save the robot arm's pose to a file."""
    pose_filename = f'{base_dir}/poses/pose_{frame_num}.npy'
    np.save(pose_filename, pose)
    print(f"Saved pose to {pose_filename}")

# %%
import sys
parent_dir = os.path.dirname(os.getcwd())
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
from realsense.realsense import Camera
from realsense.realsense import get_devices


def capture_images(camera, delay_before_shooting, start_frame, picture_nums, base_dir, init_pose, roll, pitch, yaw, z_add, x_add, y_add,
                   max_roll_deviation, max_pitch_deviation, max_yaw_deviation):
    
    camera.start()
    
    rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs = camera.get_intrinsics_raw()
    depth_scale = camera.get_depth_scale()

    print(f"RGB Intrinsics: {rgb_intrinsics}")
    print(f"RGB Distortion Coefficients: {rgb_coeffs}")
    rgb_intrinsics_path = f'{base_dir}/rgb_intrinsics.npz'
    np.savez(rgb_intrinsics_path, fx=rgb_intrinsics.fx, fy=rgb_intrinsics.fy, ppx=rgb_intrinsics.ppx, ppy=rgb_intrinsics.ppy, coeffs=rgb_intrinsics.coeffs)

    print(f"Depth Scale: {depth_scale}")
    print(f"Depth Intrinsics: {depth_intrinsics}")
    print(f"Depth Distortion Coefficients: {depth_coeffs}")
    depth_intrinsics_path = f'{base_dir}/depth_intrinsics.npz'
    np.savez(depth_intrinsics_path, fx=depth_intrinsics.fx, fy=depth_intrinsics.fy, ppx=depth_intrinsics.ppx, ppy=depth_intrinsics.ppy, coeffs=depth_intrinsics.coeffs, depth_scale=depth_scale)

    # drop the first few frames to allow the camera to warm up
    _, _ = camera.shoot()  
    time.sleep(delay_before_shooting)

    for frame_num in range(start_frame, start_frame + picture_nums):  # Capture images at 10 different poses
        pose = generate_and_move_to_pose(init_pose, roll, pitch, yaw, z_add, x_add, y_add,
                                         max_roll_deviation, max_pitch_deviation, max_yaw_deviation)
        rgb_image, depth_image = camera.shoot()
        rgb_filename = f'{base_dir}/rgb/{frame_num}.png'
        depth_filename = f'{base_dir}/depth/{frame_num}.npy'
        plt.imsave(rgb_filename, rgb_image)
        np.save(depth_filename, depth_image)
        print(f"Saved {rgb_filename}")
        print(f"Saved {depth_filename}")

        save_pose(pose, base_dir, frame_num)

    panda.move_to_start()
        
    camera.stop()


# %%
base_dir = '../../hand_in_eye2'
os.makedirs(f'{base_dir}/rgb', exist_ok=True)
os.makedirs(f'{base_dir}/depth', exist_ok=True)
os.makedirs(f'{base_dir}/poses', exist_ok=True)
# Define a list of configurations
image_configs = [
    {
        'base_dir': '../hand_in_eye2',
        'init_pose': panda_py.fk(constants.JOINT_POSITION_START),
        'roll': 0.0,
        'pitch': 0.2,
        'yaw': 0.0,
        'z_add': 0.16,
        'x_add': 0.12,
        'y_add': 0.0,
        'max_roll_deviation': 0.1,
        'max_pitch_deviation': 0.1,
        'max_yaw_deviation': 0.1
    },
    {
        'base_dir': '../hand_in_eye2',
        'init_pose': panda_py.fk(constants.JOINT_POSITION_START),
        'roll': 0.3,
        'pitch': 0.35,
        'yaw': 0.0,
        'z_add': 0.1,
        'x_add': 0.1,
        'y_add': -0.25,
        'max_roll_deviation': 0.1,
        'max_pitch_deviation': 0.1,
        'max_yaw_deviation': 0.1
    },
    {
        'base_dir': '../hand_in_eye2',
        'init_pose': panda_py.fk(constants.JOINT_POSITION_START),
        'roll': -0.25,
        'pitch': 0.35,
        'yaw': 0.2,
        'z_add': 0.12,
        'x_add': 0.1,
        'y_add': 0.20,
        'max_roll_deviation': 0.1,
        'max_pitch_deviation': 0.1,
        'max_yaw_deviation': 0.1
    },
    {
        'base_dir': '../hand_in_eye2',
        'init_pose': panda_py.fk(constants.JOINT_POSITION_START),
        'roll': -0.25,
        'pitch': 0.2,
        'yaw': 0.2,
        'z_add': 0.20,
        'x_add': 0.11,
        'y_add': 0.1,
        'max_roll_deviation': 0.1,
        'max_pitch_deviation': 0.1,
        'max_yaw_deviation': 0.1
    }
]


# Enumerate connected RealSense cameras
device_serials = get_devices()

# Print selected device serial numbers
print("Selected device serial numbers:", device_serials[0])

rgb_resolution = (1280, 720)  # RGB resolution (width, height)
depth_resolution = (1280, 720)  # Depth resolution (width, height)

camera = Camera(device_serials[0], rgb_resolution, depth_resolution)

# Delay before shooting (in seconds)
delay_before_shooting = 2.0

# Iterate over the list of configurations and capture images
for i, config in enumerate(image_configs):
    capture_images(camera, delay_before_shooting, 20*i, 20, base_dir, config['init_pose'], config['roll'], config['pitch'], config['yaw'],
                   config['z_add'], config['x_add'], config['y_add'], config['max_roll_deviation'],
                   config['max_pitch_deviation'], config['max_yaw_deviation'])

# %%
# import math

# def calculate_trajectory(center, height, radius, each_angle):
#     x_center, y_center = center
#     # Initialize lists to store coordinates
#     x_trajectory = []
#     y_trajectory = []
#     z_trajectory = []
    
#     init_pose = panda_py.fk(constants.JOINT_POSITION_START)
#     panda.move_to_start()
    
#     # Calculate trajectory points
#     for angle in range(0, 360, each_angle):
#         # Convert angle to radians
#         angle_rad = math.radians(angle)
        
#         # Calculate coordinates of point on circle
#         x_point = x_center + radius * math.cos(angle_rad)
#         y_point = y_center + radius * math.sin(angle_rad)
#         z_point = height
        
#         panda.move_to_start()
        
#         # Move the robot arm to the calculated point
#         pose = init_pose.copy()
#         pose[0, 3] = x_point
#         pose[1, 3] = y_point
#         pose[2, 3] = z_point
        
#         import roboticstoolbox as rtb
#         robot = rtb.models.Panda()
#         from spatialmath import SE3
#         import transforms3d.euler as euler

#         rotation_matrix = pose[:3, :3]
#         translation = pose[:3, 3]

#         roll, pitch, yaw = euler.mat2euler(rotation_matrix)
#         Tep = SE3.Trans(translation) * SE3.RPY([roll, pitch, yaw])
#         print(Tep)
#         sol = robot.ik_LM(Tep)         # solve IK
#         # print(sol)
        
#         panda.move_to_joint_position(sol[0])
        
        
#         # Append coordinates to trajectory lists
#         x_trajectory.append(x_point)
#         y_trajectory.append(y_point)
#         z_trajectory.append(z_point)
    
#     return x_trajectory, y_trajectory, z_trajectory

# # Example usage:
# center = (0.48, 0)
# height = 0.5
# radius = 0.17
# each_angle = 45  # Angle increment in degrees

# x_traj, y_traj, z_traj = calculate_trajectory(center, height, radius, each_angle)
# print("X trajectory:", x_traj)
# print("Y trajectory:", y_traj)
# print("Z trajectory:", z_traj)


# %%
# panda.move_to_start()
# # init_pose = panda_py.fk(constants.JOINT_POSITION_START)
# # print(init_pose)
# # panda.move_to_pose(init_pose)

# %%
# # plot the trajectory xy
# plt.plot(x_traj, y_traj)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('XY Trajectory')
# plt.show()

# %%


