import rasterio
import numpy as np

print("=== Checking Pixel Value Differences ===")

# Read first 100x100 pixels from each band
with rasterio.open('B02_10m.tif') as b02, \
     rasterio.open('B03_10m.tif') as b03, \
     rasterio.open('B04_10m.tif') as b04:
    
    # Read small sample
    sample_b02 = b02.read(1, window=((0, 100), (0, 100)))
    sample_b03 = b03.read(1, window=((0, 100), (0, 100)))
    sample_b04 = b04.read(1, window=((0, 100), (0, 100)))
    
    print(f"B02 (Blue) - Min: {sample_b02.min()}, Max: {sample_b02.max()}, Mean: {sample_b02.mean():.2f}")
    print(f"B03 (Green) - Min: {sample_b03.min()}, Max: {sample_b03.max()}, Mean: {sample_b03.mean():.2f}")
    print(f"B04 (Red) - Min: {sample_b04.min()}, Max: {sample_b04.max()}, Mean: {sample_b04.mean():.2f}")
    
    print(f"\nBands are identical: {np.array_equal(sample_b02, sample_b03) and np.array_equal(sample_b03, sample_b04)}")
