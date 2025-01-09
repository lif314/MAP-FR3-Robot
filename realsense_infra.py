import pyrealsense2 as rs
import numpy as np
import cv2

# 创建一个管道来封装实际的设备和传感器
pipeline = rs.pipeline()

# 创建一个配置对象来配置流
config = rs.config()
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.infrared, 1)  # 启用左红外流
config.enable_stream(rs.stream.infrared, 2)  # 启用右红外流

# 启动流
profile = pipeline.start(config)

# 获取相机内参
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrin = depth_profile.get_intrinsics()

# 创建点云对象
pc = rs.pointcloud()

try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        infrared_frame1 = frames.get_infrared_frame(1)
        infrared_frame2 = frames.get_infrared_frame(2)

        # 将深度帧转换为点云
        points = pc.calculate(depth_frame)
        pc.map_to(infrared_frame1)

        # 获取点云数据
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz

        # 显示深度图像
        depth_image = np.asanyarray(depth_frame.get_data())
        cv2.imshow('Depth Image', depth_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止流
    pipeline.stop()
    cv2.destroyAllWindows()