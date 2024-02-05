import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

# Step 1: Specify the path to the folder containing your video files
video_folder_path = "C:\\Users\\nkart\\Desktop\\Capstone\\ASL_Citizen\\videos"

# Step 2: Automatically list all video files in the specified folder
video_files = [os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

# Initialize an empty DataFrame to store video metadata
columns = ['File Name', 'Duration (seconds)', 'Resolution', 'Frame Rate (fps)']
video_metadata_df = pd.DataFrame(columns=columns)

# Function to extract video metadata, modified to handle batches
def extract_video_metadata_batch(video_paths):
    batch_data = []
    for video_path in video_paths:
        with VideoFileClip(video_path) as video:
            duration = video.duration
            resolution = video.size
            fps = video.fps
        file_name = os.path.basename(video_path)
        batch_data.append([file_name, duration, f"{resolution[0]}x{resolution[1]}", fps])
    return pd.DataFrame(batch_data, columns=columns)

# Process videos in batches to provide status updates and manage memory usage
batch_size = 1000  # Adjust based on your system's memory capacity and processing power
for i in range(0, len(video_files), batch_size):
    batch_files = video_files[i:i+batch_size]
    batch_metadata = extract_video_metadata_batch(batch_files)
    video_metadata_df = pd.concat([video_metadata_df, batch_metadata], ignore_index=True)
    print(f"Processed {min(i + batch_size, len(video_files))}/{len(video_files)} videos.")

# Visualization of Video Durations and Frame Rates
# Use binning for durations
plt.figure(figsize=(12, 8))
duration_bins = np.linspace(video_metadata_df['Duration (seconds)'].min(), video_metadata_df['Duration (seconds)'].max(), 50)
plt.hist(video_metadata_df['Duration (seconds)'], bins=duration_bins, color='skyblue', edgecolor='black')
plt.title('Histogram of Video Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Number of Videos')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Aggregate and visualize frame rates
plt.figure(figsize=(12, 8))
video_metadata_df['Rounded Frame Rate'] = video_metadata_df['Frame Rate (fps)'].round().astype(int)
frame_rate_counts = video_metadata_df['Rounded Frame Rate'].value_counts().sort_index()
frame_rate_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Bar Chart of Frame Rates')
plt.xlabel('Frame Rate (fps)')
plt.ylabel('Number of Videos')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.75)
plt.show()
