import os
import random
import numpy as np
import pandas as pd
import laspy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from jakteristics import compute_features, FEATURE_NAMES
from tqdm import tqdm
import shutil

def get_features(point_cloud):
    points = point_cloud.xyz
    height = point_cloud.z
    f = {}
    f['n_points'] = len(points)
    f['h_max'] = np.max(height)
    f['h_mean'] = np.mean(height)
    f['h_median'] = np.median(height)
    f['h_std'] = np.std(height)
    # Histogram of height
    hist, bin_edges = np.histogram(height, bins=5)
    hist = hist / np.sum(hist)
    for i in range(len(hist)):
        f[f'h_hist_{i}'] = hist[i]

    # Compute radial distances for horizontal distribution
    centroid = np.mean(points[:, :2], axis=0)
    radial_distances = np.linalg.norm(points[:, :2] - centroid, axis=1)
    # Histogram of radial distances
    radial_hist, radial_bin_edges = np.histogram(radial_distances, bins=5)
    radial_hist = radial_hist / np.sum(radial_hist)
    for i in range(len(radial_hist)):
        f[f'radial_hist_{i}'] = radial_hist[i]

    geo_features = compute_features(points, search_radius=5, feature_names=FEATURE_NAMES)
    geo_features = np.nan_to_num(geo_features)
    geo_histograms = [np.histogram(geo_features[:, i], bins=5)[0] for i in range(geo_features.shape[1])]
    # Normalize
    geo_histograms = [hist / np.sum(hist) for hist in geo_histograms]
    # Write to f dict with FEATURE_NAMES
    for i, feature_name in enumerate(FEATURE_NAMES):
        for j in range(len(geo_histograms[i])):
            f[f'{feature_name}_hist_{j}'] = geo_histograms[i][j]

    return f

def load_sampled_file_paths(folder_path, sample_size=500):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.las')]
    sampled_files = random.sample(all_files, sample_size)
    return sampled_files


sample_size = 500
# Load file paths
real_file_paths = load_sampled_file_paths("data/wytham_gt_pred/Real", sample_size=sample_size)
pred_file_paths = load_sampled_file_paths("data/wytham_gt_pred/Pred", sample_size=sample_size)
feature_list = []

print("Extracting features...")

for file_path in tqdm(real_file_paths):
        pc = laspy.read(file_path)
        features = get_features(pc)
        features['filename'] = file_path  # Include filename in the features
        features['label'] = 0  # 0 for Real
        feature_list.append(features)

for file_path in tqdm(pred_file_paths):
        pc = laspy.read(file_path)
        features = get_features(pc)
        features['filename'] = file_path  # Include filename in the features
        features['label'] = 1  # 1 for Pred
        feature_list.append(features)
# Convert list of dictionaries to DataFrame
df = pd.DataFrame(feature_list)

# Split into features and labels
X = df.drop(columns=['label', 'filename'])
y = df['label']
filenames = df['filename']

print("Training Random Forest Classifier...")

# Split into training and validation sets (90% train, 10% validation)
X_train, X_val, y_train, y_val, filenames_train, filenames_val = train_test_split(
    X, y, filenames, test_size=0.1, random_state=42)

# Train a Random Forest Classifier
class_weights = {0: 1, 1: 0.5} 
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights)
clf.fit(X_train, y_train)

#report
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))

# Predict probabilities on the validation set
y_prob = clf.predict_proba(X_val)[:, 1]  # Probability of being in the 'Pred' class

# Create a DataFrame for validation results
val_results = pd.DataFrame({'filename': filenames_val, 'label': y_val, 'probability': y_prob})

# Filter for 'Pred' samples (label = 1)
pred_results = val_results[val_results['label'] == 1]

# Find the samples with the highest and lowest probabilities
highest_prob = pred_results.nlargest(5, 'probability')
lowest_prob = pred_results.nsmallest(5, 'probability')

# Print the results
print("Highest Probability Pred Sample:")
print(highest_prob)

print("\nLowest Probability Pred Sample:")
print(lowest_prob)

# Save the results to a CSV file
pred_results.to_csv('pred_sample_probabilities.csv', index=False)

n_samples = 5

# Create directories for highest and lowest probability point clouds if they don't exist
highest_prob_dir = "highest_prob_samples"
lowest_prob_dir = "lowest_prob_samples"
os.makedirs(highest_prob_dir, exist_ok=True)
os.makedirs(lowest_prob_dir, exist_ok=True)

# Copy the highest probability point cloud files
highest_prob_filenames = highest_prob.nlargest(n_samples, 'probability')['filename']
for filename in highest_prob_filenames:
    shutil.copy(filename, highest_prob_dir)
print(f"Top {n_samples} highest probability point clouds copied to {highest_prob_dir}")

# Copy the lowest probability point cloud files
lowest_prob_filenames = lowest_prob.nsmallest(n_samples, 'probability')['filename']
for filename in lowest_prob_filenames:
    shutil.copy(filename, lowest_prob_dir)
print(f"Top {n_samples} lowest probability point clouds copied to {lowest_prob_dir}")