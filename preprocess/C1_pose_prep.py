import os
import cv2
import json
import mediapipe as mp
import sys

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

#  directory containing all video output 
root_dir = "/workspace/cstudent4/RGBD_IMU_Dataset/exp/engagment_data/"

# for video_folder in os.listdir(root_dir):
for video_folder in ['WIN_20241005_17_23_28_Pro','WIN_20240921_17_30_17_Pro']:#:os.listdir(root_dir):

    video_folder_path = os.path.join(root_dir, video_folder)
    if os.path.isdir(video_folder_path):
        print(f"Processing folder: {video_folder}")
        
        output_rgb = os.path.join(video_folder_path, "rgb")
        output_pose = os.path.join(video_folder_path, "pose")
        output_pose_keypoints_path = os.path.join(video_folder_path, "pose_keypoints.json")
        
        os.makedirs(output_pose, exist_ok=True)
        pose_keypoints = []
        image_files = sorted(os.listdir(output_rgb), key=lambda x: int(x.split('_')[1].replace('.jpg', '')))

        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(output_rgb, img_file)
            
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]  # pose keypoints (x, y, z)
                
                frame_number = int(img_file.split('_')[1].replace('.jpg', ''))
                pose_keypoints.append({"frame": frame_number, "keypoints": keypoints})
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                pose_overlay_img_path = os.path.join(output_pose, img_file)
                cv2.imwrite(pose_overlay_img_path, img)
            else:
                print(f"No pose detected for {img_file}")

            sys.stdout.write(f"\rProcessing frame {idx + 1}/{len(image_files)}: {img_file}")
            sys.stdout.flush()

        print()  
        with open(output_pose_keypoints_path, "w") as f:
            json.dump(pose_keypoints, f)

        print(f"\nPose keypoints saved for {video_folder} to {output_pose_keypoints_path}")




# import os
# import json

# root_dir = "/workspace/cstudent4/RGBD_IMU_Dataset/exp/engagment_data/"

# for video_folder in os.listdir(root_dir):
#     video_folder_path = os.path.join(root_dir, video_folder)
    
#     if os.path.isdir(video_folder_path):
#         print(f"Processing folder: {video_folder}")
        
#         unified_pose_json_path = os.path.join(video_folder_path, "pose_keypoints.json")
        
#         if os.path.exists(unified_pose_json_path):
#             pose_json_output_dir = os.path.join(video_folder_path, "pose_json")
#             os.makedirs(pose_json_output_dir, exist_ok=True)  
#             with open(unified_pose_json_path, 'r') as f:
#                 unified_pose_data = json.load(f)

#             for frame_data in unified_pose_data:
#                 frame_number = frame_data['frame']  
#                 keypoints = frame_data['keypoints'] 
                
#                 frame_pose_data = {
#                     'frame': frame_number,
#                     'keypoints': keypoints
#                 }
                
#                 frame_pose_json_path = os.path.join(pose_json_output_dir, f"keypoints_{frame_number}.json")
                
#                 with open(frame_pose_json_path, 'w') as f_out:
#                     json.dump(frame_pose_data, f_out)

#                 # Print progress
#                 print(f"Saved pose keypoints for frame {frame_number} to {frame_pose_json_path}")
        
#         else:
#             print(f"pose keypoints JSON not found in {video_folder_path}")

 

 