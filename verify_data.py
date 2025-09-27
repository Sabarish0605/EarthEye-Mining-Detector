import rasterio
import os

tiff_files = ['B02_10m.tif', 'B03_10m.tif', 'B04_10m.tif']

print("=== Verifying Jharia Mining TIFF Files ===")
for tiff in tiff_files:
    if os.path.exists(tiff):
        with rasterio.open(tiff) as src:
            print(f"\n{tiff}:")
            print(f"  Shape: {src.width} x {src.height}")
            print(f"  Bands: {src.count}")
            print(f"  CRS: {src.crs}")
            print(f"  Data type: {src.dtypes[0]}")
            print(f"  File size: {os.path.getsize(tiff)/(1024*1024):.2f} MB")
    else:
        print(f"{tiff}: FILE NOT FOUND")
