import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import open3d as o3d


# Specifying the folders with the datasets
FOLDER_PATHS = [
    r"C:\Users\Dell 88\Desktop\LiDar Data\Data Lidar-20250519\192.168.26.26_2020-11-25_20-01-45_frame-1899_part_1",
    r"C:\Users\Dell 88\Desktop\LiDar Data\Data Lidar-20250519\192.168.26.26_2020-11-25_20-01-45_frame-2155_part_2",
    r"C:\Users\Dell 88\Desktop\LiDar Data\Data Lidar-20250519\192.168.26.26_2020-11-25_20-01-45_frame-2414_part_3",
    r"C:\Users\Dell 88\Desktop\LiDar Data\Data Lidar-20250519\192.168.26.26_2020-11-25_20-01-45_frame-2566_part_4"
]

all_frames = []

for folder in FOLDER_PATHS:
    if not os.path.exists(folder):
        print(f"This folder was not found: {folder}")
        continue

    csv_files = sorted(glob(os.path.join(folder, '*.csv')))
    print(f"Found {len(csv_files)} files from: {folder}")
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['frame_id'] = os.path.basename(csv_file)
        all_frames.append(df)

print(f"Total loaded frames: {len(all_frames)}")