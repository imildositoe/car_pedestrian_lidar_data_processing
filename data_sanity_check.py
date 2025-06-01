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

# Finding and counting the csv files
# Counting the total found frames
all_frames = []

for folder in FOLDER_PATHS:
    if not os.path.exists(folder):
        print(f"This folder was not found: {folder}")
        continue

    csv_files = sorted(glob(os.path.join(folder, '*.csv')))
    print(f"Found {len(csv_files)} files from: {folder}")
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, delimiter=';')
        df['FRAME_ID'] = os.path.basename(csv_file)
        all_frames.append(df)

print(f"Total loaded frames: {len(all_frames)}")

# This snippet is for the coordinate distribution sanity check
distances = [df['DISTANCE'].values for df in all_frames]
distances_flat = np.concatenate(distances)
print("\nDistance Check")
print(f"Min Distance: {distances_flat.min():.2f} m")
print(f"Max Distance: {distances_flat.max():.2f} m")
print(f"Outside range (5-250m): {np.sum((distances_flat < 5) | (distances_flat > 250))} points")

# This snippet is for the temporal coherence sanity check
timestamps = [df['TIMESTAMP'].values[0] for df in all_frames]
sorted_ts = np.sort(timestamps)
gaps = np.diff(sorted_ts)
print("\nTimestamp Check")
print(f"Avg time step: {np.mean(gaps):.2e}")
print(f"Max gap: {np.max(gaps):.2e}")
print(f"Min gap: {np.min(gaps):.2e}")
if np.any(gaps > 1e6):
    print("Warning: Potential missing frames")

# This snippet is for the point cloud density sanity check
point_counts = [len(df) for df in all_frames]
print("\nPoint Cloud Density Check")
print(f"Average points per frame: {np.mean(point_counts):.0f}")
print(f"Min points: {np.min(point_counts)}")
print(f"Max points: {np.max(point_counts)}")

# This snippet is for the visualization of a sample framepoint cloud density
def visualize_frame(df):
    pcd = o3d.geometry.PointCloud()
    points = df[['X', 'Y', 'Z']].values
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.zeros((len(points), 3))
    intensity_scaled = df['INTENSITY'].values / df['INTENSITY'].max()
    colors[:, 0] = intensity_scaled
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd], window_name="FramePoint Cloud")

sample_df = all_frames[len(all_frames) // 2]
visualize_frame(sample_df)