# MITMEP

# A Multimodal Dataset for Enhancing Industrial Task Monitoring and Engagement Prediction

## Overview

This repository provides access to a unique **multimodal dataset** aimed at improving industrial task monitoring and engagement prediction. The dataset captures comprehensive interactions in industrial workflows, combining **RGBD video, IMU sensor data, and manual annotations** to support various tasks like **action recognition, engagement level prediction, and task completion verification**.

Key features of this dataset include:
- **Multimodal recordings:** RGB-D video, IMU sensor data, and hand-object interaction labels.
- **Diverse industrial actions:** Covering assembly, alignment, and operational tasks.
- **Engagement labels:** Predicted through operator motion and visual cues.
- Designed for applications in **Human-Robot Interaction (HRI)**, ergonomics studies, and task monitoring.
 
---

## Repository Structure

```plaintext
├── dataset/
│   ├── RGBD/
│   ├── IMU/
│   ├── annotations/
│   └── metadata/
├── examples/
│   ├── preprocessing/
│   ├── model_baselines/
│   └── visualization/
├── scripts/
│   ├── data_loader.py
│   ├── imu_preprocess.py
│   ├── rgbd_visualizer.py
│   └── engagement_predictor.py
├── README.md
└── LICENSE
