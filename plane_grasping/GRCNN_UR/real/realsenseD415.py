import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 创建保存目录
rgb_dir = "data/plane_grasping/rgb"
depth_dir = "data/plane_grasping/depth"
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# 初始化RealSense管道
pipeline = rs.pipeline()
config = rs.config()

# 配置流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动管道
pipeline.start(config)

# 创建对齐对象，将深度图对齐到彩色图
align_to = rs.stream.color
align = rs.align(align_to)

try:
    print("开始捕获图像，按S键保存，按ESC键退出")
    count = 0
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()

        # 对齐深度图到RGB图像
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 将深度帧和彩色帧转为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 可视化深度图
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 显示图像
        cv2.imshow('RGB Image', color_image)
        cv2.imshow('Depth Image (Aligned)', depth_colormap)

        # 检测键盘输入
        key = cv2.waitKey(1)
        if key == ord('s'):  # 按下'S'键保存图像
            rgb_path = os.path.join(rgb_dir, f"{count:04d}.png")
            depth_path = os.path.join(depth_dir, f"{count:04d}.png")
            
            cv2.imwrite(rgb_path, color_image)
            cv2.imwrite(depth_path, depth_image)

            print(f"保存图像：{rgb_path} 和 {depth_path}")
            count += 1

        elif key == 27:  # 按下ESC键退出
            print("退出程序")
            break

except KeyboardInterrupt:
    print("捕获结束")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()