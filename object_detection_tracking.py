import data_sanity_check as dsc
import pandas as pd
import numpy as np
from glob import glob
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

dsc

# Function to remove ground points, adjust, and normalize distance intensity.
def preprocess_frame(df):
    df = df[df['Z'] > -0.5].copy()
    df.loc[:, 'DISTANCE_NORM'] = StandardScaler().fit_transform(df[['DISTANCE']])
    df.loc[:, 'INTENSITY_NORM'] = StandardScaler().fit_transform(df[['INTENSITY']])
    return df[['X', 'Y', 'Z', 'DISTANCE_NORM', 'INTENSITY_NORM']]

# This snippet is to detect the objects and cluster them using DBSCAN.
def detect_objects(df, eps=1.0, min_samples=10):
    coords = df[['X', 'Y', 'Z']].values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df['cluster'] = db.labels_
    return df[df['cluster'] != -1]

def get_object_centroids(df):
    return df.groupby('cluster')[['X', 'Y', 'Z']].mean().reset_index()

all_detections = []

for i, frame in enumerate(dsc.all_frames):
    clean_frame = preprocess_frame(frame)
    clustered = detect_objects(clean_frame)
    centroids = get_object_centroids(clustered)
    centroids['frame_index'] = i
    all_detections.append(centroids)

# This snippet is responsible for defining the logic to track the detected objects using 
# the centroid-based tracking algorithm supported by the Hungarian assignment method.
tracks = []
track_id = 0
last_centroids = None

for frame_id, detections in enumerate(all_detections):
    detections = detections.copy()
    if last_centroids is None:
        detections['track_id'] = range(track_id, track_id + len(detections))
        track_id += len(detections)
    else:
        cost_matrix = np.linalg.norm(
            last_centroids[['X', 'Y']].values[:, None, :] - detections[['X', 'Y']].values[None, :, :], axis=2
        )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        detections['track_id'] = -1

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 2.0:
                detections.at[c, 'track_id'] = last_centroids.iloc[r]['track_id']

        for idx in detections[detections['track_id'] == -1].index:
            detections.at[idx, 'track_id'] = track_id
            track_id += 1

    tracks.append(detections)
    last_centroids = detections[['X', 'Y', 'Z', 'track_id']]

all_tracks_df = pd.concat(tracks, ignore_index=True)

# This snippet is to classify the objects using the custom-defined heuristic method.
def classify_object(track_df):
    size_xy = np.linalg.norm(track_df[['X', 'Y']].max() - track_df[['X', 'Y']].min())
    height = track_df['Z'].max() - track_df['Z'].min()
    duration = track_df['frame_index'].nunique()
    mean_motion = track_df[['X', 'Y']].diff().abs().mean().mean()

    if height > 3.5:
        return 'treetop'
    elif size_xy > 10 and duration > 100:
        return 'row of houses'
    elif height < 0.5 and size_xy < 1.0:
        return 'pedestrian'
    elif size_xy < 2.0:
        return 'cyclist'
    elif mean_motion < 0.01:
        return 'parked car'
    else:
        return 'vehicle'

track_labels = []
for tid, track in all_tracks_df.groupby('track_id'):
    label = classify_object(track)
    all_tracks_df.loc[all_tracks_df['track_id'] == tid, 'object_type'] = label

# This snippet is responsible for plotting the resulting tracked objects
plt.figure(figsize=(10, 6))
for obj_type in all_tracks_df['object_type'].unique():
    subset = all_tracks_df[all_tracks_df['object_type'] == obj_type]
    plt.scatter(subset['X'], subset['Y'], label=obj_type, alpha=0.6)
plt.legend()
plt.title("Classified Objects Over Time")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()


# Block to evaluate the performance of the algorithm.
# It was used the y_true generated randomly, which emulates the labeled ground truth.
y_true = np.random.choice(
    ['pedestrian', 'cyclist', 'parked car', 'vehicle', 'row of houses', 'treetop'], 
    size=len(all_tracks_df)
)
y_pred = all_tracks_df['object_type']

# Printing the classification report.
print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

# Snippet to plot the Confusion Matrix.
labels = ['pedestrian', 'cyclist', 'parked car', 'vehicle', 'row of houses', 'treetop']
conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
sns.heatmap(conf_matrix, annot=True, xticklabels=labels, yticklabels=labels, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Snippet to compute and print the False Alarm Rate.
false_positives = (y_pred != y_true).sum()
false_alarm_rate = false_positives / 7200
print(f"False alarm rate: {false_alarm_rate:.5f} per hour")

# Snippet to compute and print the correct classification rate.
correct_predictions = (y_pred == y_true).sum()
total_predictions = len(y_true)
accuracy = correct_predictions / total_predictions
print(f"Correct classification rate: {accuracy:.5f}")

# Snippet to print the overall evaluation summary.
print("\nOverall Evaluation Summary:")
print(f"Total objects evaluated: {total_predictions}")
print(f"Correctly classified: {correct_predictions}")
print(f"False positives: {false_positives}")