import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cv2

print("=== EarthEye Complete Mining Detection Algorithm ===")

# Load the three bands
with rasterio.open('B02_10m.tif') as b02, \
     rasterio.open('B03_10m.tif') as b03, \
     rasterio.open('B04_10m.tif') as b04:
    
    blue = b02.read(1).astype(np.float32)
    green = b03.read(1).astype(np.float32)
    red = b04.read(1).astype(np.float32)
    
    # Get metadata for saving results
    profile = b02.profile

print(f"Loaded bands: {blue.shape}")

# Calculate spectral indices
def calculate_mining_indices(blue, green, red):
    print("Calculating spectral indices...")
    
    # Avoid division by zero
    epsilon = 1e-10
    
    # NDBI - Normalized Difference Bare Index
    ndbi = (red - green) / (red + green + epsilon)
    
    # Bare Soil Index
    bare_soil = (red + blue - green) / (red + blue + green + epsilon)
    
    # Brightness
    brightness = (blue + green + red) / 3
    
    # Enhanced Mining Index (custom)
    mining_index = (red * blue) / (green * green + epsilon)
    
    return ndbi, bare_soil, brightness, mining_index

ndbi, bare_soil, brightness, mining_index = calculate_mining_indices(blue, green, red)

# Prepare data for classification
def prepare_classification_data():
    print("Preparing classification data...")
    
    # Stack all features
    features = np.stack([blue, green, red, ndbi, bare_soil, brightness, mining_index], axis=-1)
    
    # Reshape for sklearn
    h, w, bands = features.shape
    features_flat = features.reshape(h*w, bands)
    
    # Remove invalid pixels (zeros, NaNs)
    valid_mask = ~np.any((features_flat == 0) | np.isnan(features_flat) | np.isinf(features_flat), axis=1)
    features_clean = features_flat[valid_mask]
    
    return features_clean, valid_mask, h, w

features_clean, valid_mask, h, w = prepare_classification_data()
print(f"Valid pixels for classification: {len(features_clean):,} / {h*w:,}")

# Perform clustering to identify land cover types
def perform_mining_classification(features, n_clusters=6):
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    return labels, kmeans, scaler

labels, kmeans, scaler = perform_mining_classification(features_clean)

# Map results back to image
def create_classification_map(labels, valid_mask, h, w):
    print("Creating classification map...")
    
    # Initialize classification image
    classification = np.zeros(h*w, dtype=np.uint8)
    classification[valid_mask] = labels + 1  # +1 to avoid 0 (background)
    classification = classification.reshape(h, w)
    
    return classification

classification_map = create_classification_map(labels, valid_mask, h, w)

# Identify potential mining areas
def identify_mining_clusters(features_clean, labels, n_clusters):
    print("Analyzing clusters to identify mining areas...")
    
    cluster_stats = {}
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_features = features_clean[cluster_mask]
        
        if len(cluster_features) > 0:
            cluster_stats[i] = {
                'count': len(cluster_features),
                'brightness_mean': np.mean(cluster_features[:, 5]),  # brightness
                'ndbi_mean': np.mean(cluster_features[:, 3]),        # ndbi
                'mining_index_mean': np.mean(cluster_features[:, 6]) # mining_index
            }
    
    # Sort clusters by mining potential (high brightness + high NDBI)
    mining_scores = {}
    for cluster_id, stats in cluster_stats.items():
        # Mining score based on brightness and NDBI
        score = stats['brightness_mean'] * 0.4 + stats['ndbi_mean'] * 0.6
        mining_scores[cluster_id] = score
        
        print(f"Cluster {cluster_id}: {stats['count']:,} pixels, "
              f"Brightness: {stats['brightness_mean']:.2f}, "
              f"NDBI: {stats['ndbi_mean']:.3f}, "
              f"Mining Score: {score:.3f}")
    
    # Identify top mining clusters
    sorted_clusters = sorted(mining_scores.items(), key=lambda x: x[1], reverse=True)
    top_mining_clusters = [cluster_id for cluster_id, score in sorted_clusters[:2]]
    
    return top_mining_clusters, cluster_stats

top_mining_clusters, cluster_stats = identify_mining_clusters(features_clean, labels, 6)
print(f"Top mining clusters: {top_mining_clusters}")

# Create final mining detection map
def create_mining_map(classification_map, top_mining_clusters):
    print("Creating final mining detection map...")
    
    mining_map = np.zeros_like(classification_map)
    for cluster_id in top_mining_clusters:
        mining_map[classification_map == (cluster_id + 1)] = 1
    
    return mining_map

mining_map = create_mining_map(classification_map, top_mining_clusters)

# Calculate mining statistics
mining_pixels = np.sum(mining_map)
total_pixels = np.sum(mining_map >= 0)
mining_percentage = (mining_pixels / total_pixels) * 100

print(f"\n=== MINING DETECTION RESULTS ===")
print(f"Total mining pixels detected: {mining_pixels:,}")
print(f"Mining area percentage: {mining_percentage:.2f}%")
print(f"Estimated mining area: {mining_pixels * 0.01:.2f} hectares")  # 10m pixels = 100m²

# Save results
profile.update(dtype=rasterio.uint8, count=1)

with rasterio.open('mining_classification.tif', 'w', **profile) as dst:
    dst.write(classification_map.astype(np.uint8), 1)

with rasterio.open('mining_detection.tif', 'w', **profile) as dst:
    dst.write(mining_map.astype(np.uint8), 1)

print("\n✅ Results saved:")
print("- mining_classification.tif (all land cover classes)")
print("- mining_detection.tif (binary mining areas)")
print("\n🎉 Phase 1 Mining Detection COMPLETED!")
