import open3d as o3d
import numpy as np
import pyrealsense2 as rs

# 创建管道
pipeline = rs.pipeline()
# 创建配置
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动管道
pipeline.start(config)

# 获取相机内参
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrin = depth_profile.as_video_stream_profile().get_intrinsics()
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrin = color_profile.as_video_stream_profile().get_intrinsics()

# 创建Open3D的点云对象
pcd = o3d.geometry.PointCloud()

# 创建Open3D的可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window()

# 获取相机内参
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=color_intrin.width,
    height=color_intrin.height,
    fx=color_intrin.fx,
    fy=color_intrin.fy,
    cx=color_intrin.ppx,
    cy=color_intrin.ppy
)

try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # 将深度图像和RGB图像转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 创建RGBD图像
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image),
            depth_scale=4000.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )
        
        # 生成点云
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsic
        )
        
        # 翻转点云，以便正确显示
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        # 更新点云
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.clear_geometries()
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止管道
    pipeline.stop()
    vis.destroy_window()