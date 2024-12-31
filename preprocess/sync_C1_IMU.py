import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d
import json
from pyts.image import GramianAngularField
import threading

#Rename all pose.json and imu.csv files to match the corresponding C1 video file names in .mp4 format, and store them in the c1_imu_data folder.

root_dir = "/workspace/cstudent4/RGBD_IMU_Dataset/exp/c1_imu_data/"

output_root_dir = "/workspace/cstudent4/RGBD_IMU_Dataset/exp/engagment_data/"



def time_diff_in_seconds(time1, time2):

    # time1 = time1.replace(microsecond=0)
    # time2 = time2.replace(microsecond=0)
    
    delta = datetime.combine(datetime.min, time1) - datetime.combine(datetime.min, time2)
    return delta.total_seconds()


def crop_imu_data(imu_data, timestamp, window=5): 
    start_time = timestamp -  window
    end_time = timestamp 
    return imu_data[(imu_data['relative_time'] >= start_time) & (imu_data['relative_time'] <= end_time)]
 
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * 2 - 1


def calculate_magnitude(df_device):
    accel_mag =  np.sqrt(df_device['Acceleration X(g)'].values**2 + 
                   df_device['Acceleration Y(g)'].values**2 + 
                   df_device['Acceleration Z(g)'].values**2)

    accel_norm = normalize_data(accel_mag)
    return accel_norm


def calculate_magnitude_comb(df_device):

        # Extract and compute GAF for Device 1
    accel_mag = np.sqrt(
        df_device['Acceleration X(g)'].values**2 +
        df_device['Acceleration Y(g)'].values**2 +
        df_device['Acceleration Z(g)'].values**2
    )
    gyro_mag = np.sqrt(
        df_device['Angular velocity X(°/s)'].values**2 +
        df_device['Angular velocity Y(°/s)'].values**2 +
        df_device['Angular velocity Z(°/s)'].values**2
    )
    angle_mag = np.sqrt(
        df_device['Angle X(°)'].values**2 +
        df_device['Angle Y(°)'].values**2 +
        df_device['Angle Z(°)'].values**2
    )
    
    # Normalize
    accel_norm = normalize_data(accel_mag)
    gyro_norm = normalize_data(gyro_mag)
    angle_norm = normalize_data(angle_mag)
    return accel_norm, gyro_norm, angle_norm



def compute_gaf(R, frame_count,target_size=(64, 64),hand = 'right'):
    gaf = GramianAngularField(method='difference')
    R_gaf = gaf.fit_transform(R.reshape(1, -1)) 
    R_gaf_image = R_gaf[0]  
    
    R_gaf_image = cv2.normalize(R_gaf_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    R_gaf_image_colored = cv2.applyColorMap(R_gaf_image, cv2.COLORMAP_JET)
    R_gaf_image_colored = cv2.resize(R_gaf_image_colored, target_size, interpolation=cv2.INTER_LINEAR)

    # output_filename = os.path.join(imu_output_gaf, f'gaf_frame_{frame_count:06d}.jpg')

    if hand == 'right':
        output_filename = os.path.join(imu_output_gaf_right, f'gaf_frame_{frame_count:06d}.jpg')
    else: 
        output_filename = os.path.join(imu_output_gaf_left, f'gaf_frame_{frame_count:06d}.jpg')

    cv2.imwrite(output_filename, R_gaf_image_colored)


def compute_and_merge_gaf(acceleration, angular_velocity, angles, frame_count, target_size=(64, 64),hand = 'right'):
    gaf = GramianAngularField(method='difference')
    
    gaf_acc = gaf.fit_transform(acceleration.reshape(1, -1))[0]  # GAF for acceleration
    gaf_ang_vel = gaf.fit_transform(angular_velocity.reshape(1, -1))[0]  # GAF for angular velocity
    gaf_angle = gaf.fit_transform(angles.reshape(1, -1))[0]  # GAF for angles
    
    gaf_acc = cv2.normalize(gaf_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gaf_ang_vel = cv2.normalize(gaf_ang_vel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gaf_angle = cv2.normalize(gaf_angle, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    gaf_acc_resized = cv2.resize(gaf_acc, target_size, interpolation=cv2.INTER_LINEAR)
    gaf_ang_vel_resized = cv2.resize(gaf_ang_vel, target_size, interpolation=cv2.INTER_LINEAR)
    gaf_angle_resized = cv2.resize(gaf_angle, target_size, interpolation=cv2.INTER_LINEAR)
    
    # merge the three GAF images into a 3-channel image (RGB)
    merged_gaf = cv2.merge([gaf_acc_resized, gaf_ang_vel_resized, gaf_angle_resized])
    
    if hand == 'right':
        output_filename = os.path.join(imu_output_gaf_merged_right, f'merged_gaf_frame_{frame_count:06d}.jpg')
    else: 
        output_filename = os.path.join(imu_output_gaf_merged_left, f'merged_gaf_frame_{frame_count:06d}.jpg')

    cv2.imwrite(output_filename, merged_gaf)




def check_label(time, engaged_segments, disengaged_segments):
    for seg in disengaged_segments:
        if seg[0] <= time <= seg[1]:
            return 'disengaged'
    for seg in engaged_segments:
        if seg[0] <= time <= seg[1]:
            return 'engaged'
    return 'unknown'




if not os.path.exists(output_root_dir):
    os.makedirs(output_root_dir)


for filename in os.listdir(root_dir):
    if filename.endswith(".mp4"):

    # if filename in ['WIN_20240925_17_20_29_Pro.mp4','WIN_20240925_16_58_07_Pro.mp4']:
        base_filename = os.path.splitext(filename)[0]
        
        video_file_path = os.path.join(root_dir, base_filename + ".mp4")
        json_file_path = os.path.join(root_dir, base_filename + ".json")
        imu_file_path = os.path.join(root_dir, base_filename + ".csv")

        if os.path.exists(json_file_path) and os.path.exists(imu_file_path):
            print("New case...",os.path.exists(imu_file_path))

            output_dir = os.path.join(output_root_dir, base_filename)
            imu_output_plots = os.path.join(output_dir, "imu_output_plots/")
            imu_output_gaf_right = os.path.join(output_dir, "imu_gaf_right/")
            imu_output_gaf_left = os.path.join(output_dir, "imu_gaf_left/")
            imu_output_gaf_merged_left = os.path.join(output_dir, "imu_gaf_merged_left/")
            imu_output_gaf_merged_right = os.path.join(output_dir, "imu_gaf_merged_right/")
            output_rgb = os.path.join(output_dir,"rgb")
            imu_output_plots_acc = os.path.join(output_dir, "imu_output_plots_acc/")

            os.makedirs(imu_output_gaf_right, exist_ok=True)
            os.makedirs(imu_output_gaf_left, exist_ok=True)
            os.makedirs(imu_output_gaf_merged_left, exist_ok=True)
            os.makedirs(imu_output_gaf_merged_right, exist_ok=True)
            os.makedirs(imu_output_plots, exist_ok=True)
            os.makedirs(output_rgb, exist_ok=True)

            os.makedirs(imu_output_plots_acc, exist_ok=True)
        else:
            print("skip case...", os.path.exists(imu_file_path))
            pass

        imu_data = pd.read_csv(imu_file_path, sep=',', index_col=False)

        df_device1 = imu_data[imu_data['Device name'] == 'd3:a2:b7:99:b3:53']  # Right hand
        df_device2 = imu_data[imu_data['Device name'] == 'cc:59:6a:eb:08:f5']  # Left hand


        video_filename = os.path.basename(video_file_path)
        time_part = video_filename[13:21]  # Extracts '18_02_42' (time part)

        video_start_time = datetime.strptime(time_part, '%H_%M_%S').time()

        print(f"Video start time (as pure time): {video_start_time}")

        # remove  whitespaces in 'Time' 
        df_device1['Time'] = df_device1['Time'].str.strip()
        df_device2['Time'] = df_device2['Time'].str.strip()

        # Convert IMU timestamps to datetime objects for calculation purposes 
        df_device1['Time_dt'] = pd.to_datetime(df_device1['Time'], format='%H:%M:%S.%f')
        df_device2['Time_dt'] = pd.to_datetime(df_device2['Time'], format='%H:%M:%S.%f')

        # time only (without date)
        imu_start_time_device1 = df_device1['Time_dt'].iloc[0].time()
        imu_start_time_device2 = df_device2['Time_dt'].iloc[0].time()

        # adjust the start time for both IMU devices and video
        max_start_time = max(video_start_time, imu_start_time_device1, imu_start_time_device2)
        print(f"Max start time: {max_start_time}")




        #  relative time in seconds from the max start time for both hands
        df_device1['relative_time'] = df_device1['Time_dt'].apply(lambda x: time_diff_in_seconds(x.time(), max_start_time))
        df_device2['relative_time'] = df_device2['Time_dt'].apply(lambda x: time_diff_in_seconds(x.time(), max_start_time))

        #  keep only rows where relative_time >= 0 
        df_device1_synced = df_device1[df_device1['relative_time'] >= 0].copy()
        df_device2_synced = df_device2[df_device2['relative_time'] >= 0].copy()

        # drop the temporary 'Time_dt' columns but keep the original 'Time' column
        df_device1_synced.drop(columns=['Time_dt'], inplace=True)
        df_device2_synced.drop(columns=['Time_dt'], inplace=True)


        device1_synced_csv = os.path.join(output_dir,'device1_synced.csv')
        device2_synced_csv = os.path.join(output_dir,'device2_synced.csv')


        # Save the synchronized data for both devices to CSV
        df_device1_synced.to_csv(device1_synced_csv, index=False)
        df_device2_synced.to_csv(device2_synced_csv, index=False)

        print(f"Device 1 synchronized data saved to: {device1_synced_csv}")
        print(f"Device 2 synchronized data saved to: {device2_synced_csv}")
        print("Final df_device1_synced (with all original columns and relative_time):")
        print(df_device1_synced.head())

        print("Final df_device2_synced (with all original columns and relative_time):")
        print(df_device2_synced.head())



        # Extract engagement and disengagement annotations from the JSON file
        with open(json_file_path, 'r') as f:
            video_annotation = json.load(f)

        engaged_segments = []
        disengaged_segments = []
        statuses = {}

        for key, value in video_annotation['metadata'].items():
            label = video_annotation['metadata'][key]['av']  # Extract the 'av' field which contains 'Engaged' or 'Disengaged'
            time_segment = video_annotation['metadata'][key]['z']
            status = video_annotation['metadata'][key].get('flg', None)  # Extract the status field ('flg')

            if 'Engaged' in label.values():
                engaged_segments.append(tuple(time_segment))
            elif 'Disengaged' in label.values():
                disengaged_segments.append(tuple(time_segment))

        cap = cv2.VideoCapture(video_file_path)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 3#int(fps / 10)  

        frame_count = 1  #  frame numbering  1
        imu_processing_start_time = 5  # processing IMU after 5 seconds of the video
        imu_save_interval = 1  # Save IMU data every 1 second 
        frame_times = []
        frame_filenames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time_in_seconds = frame_count / 30

            if df_device1_synced['relative_time'].max()<current_time_in_seconds or df_device2_synced['relative_time'].max()<current_time_in_seconds:
                break 

            # sample frames at a 10 FPS rate
            if frame_count % frame_interval == 0:
                # Save frame with actual frame index

                output_filename = os.path.join(output_rgb, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(output_filename, frame)
                
                frame_times.append(current_time_in_seconds)
                frame_filenames.append(f"frame_{frame_count:06d}.jpg")
                
                # start IMU processing after the first 5 seconds of the RGB video
                if current_time_in_seconds >= imu_processing_start_time:
                    print("Processing IMU for time:", current_time_in_seconds)
                    try:
                        # Crop IMU data for both right and left hand for a 5-second window
                        imu_data_device1 = crop_imu_data(df_device1_synced, imu_processing_start_time)
                        imu_data_device2 = crop_imu_data(df_device2_synced, imu_processing_start_time)
                        
                        # calculate magnitude and resample
                        R1 = calculate_magnitude(imu_data_device1)
                        timestamps_device1 = imu_data_device1['relative_time']
                        # R1_resampled, timestamps_device1_resampled = R1.values, timestamps_device1.values#resample_to_50_points(R1.values, timestamps_device1.values)

                        R2 = calculate_magnitude(imu_data_device2)
                        timestamps_device2 = imu_data_device2['relative_time']
                        # R2_resampled, timestamps_device2_resampled = R2.values, timestamps_device2.values#resample_to_50_points(R2.values, timestamps_device2.values)
                        
                        # Plot and save IMU data
                        plt.figure(figsize=(12, 16))
                        plt.subplot(2, 1, 1)
                        plt.plot(timestamps_device1, R1, label='Accel Magnitude - Device 1 (Right)', color='brown')
                        plt.xlabel('Time (s)')
                        plt.ylabel('Acceleration (g)')
                        plt.title(f'IMU Data for Right Hand - Frame {frame_count}')  
                        plt.legend()

                        plt.subplot(2, 1, 2)
                        plt.plot(timestamps_device2, R2, label='Accel Magnitude - Device 2 (Left)', color='brown')
                        plt.xlabel('Time (s)')
                        plt.ylabel('Acceleration (g)')
                        plt.title(f'IMU Data for Left Hand - Frame {frame_count}')  
                        plt.legend()

                        plt.tight_layout()
                        plt.savefig(os.path.join(imu_output_plots_acc, f"imu_frame_{frame_count:06d}.jpg"))  
                        plt.close()


                        compute_gaf(R1, frame_count,hand = 'right')
                        compute_gaf(R2, frame_count,hand = 'left')

                        acceleration1, angular_velocity1, angles1 = calculate_magnitude_comb(imu_data_device1)
                        timestamps_device1 = imu_data_device1['relative_time']

                        compute_and_merge_gaf(acceleration1, angular_velocity1, angles1, frame_count,hand = 'right')

                        acceleration2, angular_velocity2, angles2 = calculate_magnitude_comb(imu_data_device2)
                        timestamps_device2 = imu_data_device2['relative_time']
                        compute_and_merge_gaf(acceleration2, angular_velocity2, angles2, frame_count,hand = 'left')

                        # plot and save IMU data
                        plt.figure(figsize=(12, 16))

                        # Right Hand
                        plt.subplot(2, 1, 1)
                        plt.plot(timestamps_device1, acceleration1, label='Accel Magnitude (Right)', color='r')
                        plt.plot(timestamps_device1, angular_velocity1, label='Angular Velocity Magnitude (Right)', color='g')
                        plt.plot(timestamps_device1, angles1, label='Angle (Right)', color='b')

                        plt.xlabel('Time (s)')
                        plt.ylabel('IMU Magnitude\n(Accel: g, Angular Vel: °/s, Angle: °)')  
                        plt.title(f'IMU Data for Right Hand - Frame {frame_count}')   
                        plt.legend()

                        # Left Hand
                        plt.subplot(2, 1, 2)
                        plt.plot(timestamps_device2, acceleration2, label='Accel Magnitude (Left)', color='r')
                        plt.plot(timestamps_device2, angular_velocity2, label='Angular Velocity Magnitude (Left)', color='g')
                        plt.plot(timestamps_device2, angles2, label='Angle (Left)', color='b')

                        plt.xlabel('Time (s)')
                        plt.ylabel('IMU Magnitude\n(Accel: g, Angular Vel: °/s, Angle: °)')   
                        plt.title(f'IMU Data for Left Hand - Frame {frame_count}')  
                        plt.legend()

                        plt.tight_layout()
                        plt.savefig(os.path.join(imu_output_plots, f"imu_frame_{frame_count:06d}.jpg")) 
                        plt.close()

                        imu_processing_start_time += imu_save_interval
                    except:
                        imu_processing_start_time += 2*imu_save_interval
                        print("Except case ...",imu_processing_start_time)
                        pass

                   

            
            frame_count += 1

        cap.release()


        #map frame times to labels (engagement/disengagement)
        frame_labels = [check_label(time, engaged_segments, disengaged_segments) for time in frame_times]
        frame_metadata = pd.DataFrame({
            'frame_filename': frame_filenames,   
            'label': frame_labels
        })

 
        eng_path = os.path.join(output_dir,'frame_metadata_with_status.csv')
        frame_metadata.to_csv(eng_path, index=False)

        print(frame_metadata.head())
