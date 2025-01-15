import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

# from FR3_Robot import FR3_Robot

from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp
import cv2
from PIL import Image
import torchvision.transforms as transforms

from real.realsenseD415 import Camera
import time
from spatialmath.base import rpy2r, r2q

import panda_py
from panda_py import libfranka


# 将相机坐标系下的物体坐标转移至机械臂底座坐标系下
def transform_camera_to_base(camera_xyz, hand_eye_matrix):
    # Append 1 to the camera coordinates to make them homogeneous
    camera_xyz_homogeneous = np.append(camera_xyz, 1.0)
 
    # Transform camera coordinates to arm coordinates using hand-eye matrix
    base_xyz_homogeneous = np.dot(hand_eye_matrix, camera_xyz_homogeneous)
 
    # Remove the homogeneous component and return arm coordinates
    base_xyz = base_xyz_homogeneous[:3]
    return base_xyz

# 将rpy转化为四元数
def rpy_to_quaternion(rpy):
    """
    Convert Roll-Pitch-Yaw (RPY) angles to quaternion.

    Parameters:
    - roll: Roll angle in degrees
    - pitch: Pitch angle in degrees
    - yaw: Yaw angle in degrees

    Returns:
    - Quaternion as a numpy array [w, x, y, z]
    """
    r = rpy2r(rpy)
    # Must be ‘sxyz’ or ‘xyzs’. Defaults to ‘sxyz’.
    q = r2q(r)

    return q


class PlaneGraspFR3:
    def __init__(self, saved_model_path=None,use_cuda=True,visualize=False,include_rgb=True,include_depth=True,output_size=300):
        if saved_model_path == None:
            saved_model_path = 'trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93'
        self.model = torch.load(saved_model_path)
        self.device = "cuda:0" if use_cuda else "cpu"
        self.visualize = visualize

        # 相机数据处理
        self.cam_data = CameraData(include_rgb=include_rgb,include_depth=include_depth,output_size=output_size)
        # 相机
        self.camera = Camera()
        self.intrinsic = self.camera.intrinsics

        # ROBOT
        self.hostname = "192.168.10.1" # Your Panda IP or hostname
        self.panda = panda_py.Panda(self.hostname)
        self.panda.set_default_behavior() # Panda default collision thresholds
        self.panda.move_to_start()
        self.orientation = self.panda.get_orientation()
        self.joint_speed_factor = 0.05
        self.cart_speed_factor = 0.05
        self.stiffness=np.array([600., 600., 600., 600., 250., 150., 50.])
        self.damping=np.array([50., 50., 50., 20., 20., 20., 10.])
        self.dq_threshold=0.001
        self.success_threshold=0.01

        # Optionally set higher thresholds to execute fast trajectories
        self.panda.get_robot().set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

        self.gripper = libfranka.Gripper(self.hostname)
        
        # home
        self.home_pos = np.array([0.29773711, 0.31993458, 0.43165937])
        self.home_orien = np.array([ 0.91185129,  0.40863425, -0.00820174,  0.0384447 ])
        
        # eye-to-hand标定结果
        self.cam_to_base_pose = np.array([[ 0.05096545 , 0.98781948, -0.14702111,  0.5748499 ],
                                        [ 0.99861288, -0.04845667,  0.02059779, -0.06987559],
                                        [ 0.01322274, -0.14786695, -0.98891887,  0.84245153],
                                        [ 0.,          0.,          0.,          1.       ]])
        
        # offset 来自标定的误差
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.z_offset = -0.005
        self.angle_offset = 3.0
       
        self.cam_depth_scale = 1.007080078124112689e-03 # 深度尺度 大概深度图/1000 表示真实深度

        if self.visualize:
            self.fig = plt.figure(figsize=(6, 6))
        else:
            self.fig = None

    def go_home(self, speed_factor=0.05):
        return self.panda.move_to_pose(position=self.home_pos, orientation=self.home_orien,
                                speed_factor=speed_factor,
                                stiffness=np.array([600., 600., 600., 600., 250., 150., 50.]),
                                damping=np.array([50., 50., 50., 20., 20., 20., 10.]),
                                dq_threshold=0.001,
                                success_threshold=0.10)
        
    def go_target(self, target_pos, target_orien, speed_factor=0.02):
        return self.panda.move_to_pose(position=target_pos, orientation=target_orien,
                        speed_factor=speed_factor,
                        stiffness=np.array([600., 600., 600., 600., 250., 150., 50.]),
                        damping=np.array([50., 50., 50., 20., 20., 20., 10.]),
                        dq_threshold=0.001,
                        success_threshold=0.10)


    def generate(self):
        ## if you want to use RGBD from camera,use me
        # Get RGB-D image from camera
        rgb, depth = self.camera.get_data() # meter
        depth = depth * self.cam_depth_scale
        depth[depth >1.2] = 0 # distance > 1.2m ,remove it

        # depth= np.expand_dims(depth, axis=2)
        x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)
        rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

        grasps = detect_grasps(q_img, ang_img, width_img)
        if len(grasps) ==0:
            print("Detect 0 grasp pose!")
            if self.visualize:
                plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, save=True)
            return False

        ## For real robot
        # Get grasp position from model output
        pos_z = depth[grasps[0].center[0] + self.cam_data.top_left[0], grasps[0].center[1] + self.cam_data.top_left[1]]
        pos_x = np.multiply(grasps[0].center[1] + self.cam_data.top_left[1] - self.intrinsic[0][2],
                            pos_z / self.intrinsic[0][0])
        pos_y = np.multiply(grasps[0].center[0] + self.cam_data.top_left[0] - self.intrinsic[1][2],
                            pos_z / self.intrinsic[1][1])
        if pos_z == 0:
            return False
        
        camera_xyz = np.asarray([pos_x, pos_y, pos_z]).reshape(3, 1)
        print("camera_xyz: ", camera_xyz)
        
        # Convert camera to robot coordinates
        base_xyz = transform_camera_to_base(camera_xyz, self.cam_to_base_pose)
        base_xyz[0] += self.x_offset
        base_xyz[1] += self.y_offset
        base_xyz[2] += self.z_offset
        print("base_xyz: ", base_xyz)
        
        # # TODO Convert camera to robot angle
        # angle = np.asarray([0, 0, grasps[0].angle])
        # angle.shape = (3, 1)
        # # rpy
        # target_angle = np.dot(self.cam_to_base_pose[0:3, 0:3], angle)
        # # 将rpy转换为四元数
        # # print("target_angle: ", target_angle)
        # yaw = target_angle[2][0] + self.angle_offset
        # print("yaw: ", yaw)
        # rpy = np.asarray([0, 0, yaw])
        # print("rpy: ", rpy)

        # quaternion = rpy_to_quaternion(rpy)
        
        # # compute gripper width
        width = grasps[0].length # mm
        if width < 25:    # detect error
            width = 0.035  # m
        elif width < 40:
            width = 0.045 # m
        elif width > 85:
            width = 0.077 # m

        # # np.save(self.grasp_pose, grasp_pose)
        if self.visualize:
            plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, save=False)
        
        # print("orientation: ", self.orientation)
        # print("quaternion:", quaternion)
        # 执行抓取
        finished = self.go_target(target_pos=base_xyz, target_orien=self.orientation, speed_factor=0.02)
        time.sleep(1) # 观察精度
        if finished:
            print("finished: ", finished)
            if self.gripper.grasp(width=0.01, speed=0.2, force=5, epsilon_inner=0.04, epsilon_outer=0.04):
                if self.panda.move_to_start():
                    if self.go_home(speed_factor=0.2):
                        if self.gripper.move(0.07, 0.2): # 释放物体
                            self.panda.move_to_start()
                            return True
            else:
                print("抓取失败!!!")
                self.panda.move_to_start()
                return False
        else:
            self.panda.move_to_start()
            return False

if __name__ == '__main__':
    g = PlaneGraspFR3(
        saved_model_path='rained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93',
        # saved_model_path='trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_42_iou_0.93',
        # saved_model_path='trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_35_iou_0.92',
        visualize=True,
        include_rgb=True
    )
    g.generate()
