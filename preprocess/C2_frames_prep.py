# Code to extract .bag file into RGB and Depth frames 
import pyrealsense2 as rs
import numpy as np
import cv2
import os

def extract_frames(bag_file):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False) 
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)  # RGB stream (30 FPS)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)   # Depth stream (30 FPS)
    pipeline.start(config)
    profile = pipeline.get_active_profile()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    # print("color_profile,depth_profile":color_profile, depth_profile)

    root_folder = "/workspace/cstudent4/RGBD_IMU_Dataset/20240920_182659"
    rgb_folder = os.path.join(root_folder, "rgb_frames")
    depth_folder = os.path.join(root_folder, "depth_frames")
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)

    # reduce fps from 30 to 10
    skip_frames = 2
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)   

    try:
        frame_count = 0
        saved_frame_count = 1  

        while True:
            frames = pipeline.wait_for_frames()
            if playback.current_status() == rs.playback_status.stopped:
                print("Reached end of .bag file.")
                break
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue
            frame_count += 1

            if (frame_count % (skip_frames + 1)) != 0:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            rgb_filename = f"{rgb_folder}/rgb_frame_{frame_count:06d}.jpg"
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(rgb_filename, image_rgb)

            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
            # depth_min = np.min(depth_image)  # Find the minimum depth value
            # depth_max = np.max(depth_image)  # Find the maximum depth value

            # if depth_max - depth_min > 0:
            #     depth_normalized = np.uint8(255 * (depth_image - depth_min) / (depth_max - depth_min))
            # else:
            #     depth_normalized = np.uint8(depth_image) 

            depth_filename = f"{depth_folder}/depth_frame_{frame_count:06d}.jpg"
            cv2.imwrite(depth_filename, depth_colormap)

            saved_frame_count += 1  

    finally:
        pipeline.stop()

# path to .bag file
bag_file = "/workspace/cstudent4/RGBD_IMU_Dataset/20240920_182659.bag"
extract_frames(bag_file)

