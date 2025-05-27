import geopandas as gpd
import rasterio
import pandas as pd
import os

# Define file paths
geojson_path = 'data/vector/gcps.geojson'
dem_path = '/Users/kdoherty/lidar_source/data/raster/processed/ellipsoidal_dem.tif'
output_dir = 'planning'
output_csv_path = os.path.join(output_dir, 'emlid_init_gcps.csv')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 1. Load gcps.geojson
print(f"Loading GeoJSON from {geojson_path}...")
try:
    gdf = gpd.read_file(geojson_path)
except Exception as e:
    print(f"Error loading GeoJSON: {e}")
    exit()

print(f"Loaded {len(gdf)} features.")
print(f"Original CRS: {gdf.crs}")

# 2. Create 'name' column
print("Creating 'name' column...")
gdf['name'] = gdf['site'] + '-' + gdf['position']

# 3. Extract ellipsoidal height from DEM
print(f"Extracting elevation from {dem_path}...")
elevations = []
try:
    with rasterio.open(dem_path) as src:
        print(f"DEM CRS: {src.crs}")
        # Ensure the DEM CRS matches the GeoDataFrame CRS before sampling
        if gdf.crs != src.crs:
            print(f"Reprojecting GeoDataFrame from {gdf.crs} to {src.crs} for elevation sampling...")
            gdf_dem_crs = gdf.to_crs(src.crs)
        else:
            gdf_dem_crs = gdf

        coordinates = [(point.x, point.y) for point in gdf_dem_crs.geometry]
        
        for val in src.sample(coordinates):
            # rasterio.sample returns an array, take the first element
            elevations.append(val[0])
except Exception as e:
    print(f"Error extracting elevation: {e}")
    exit()

gdf['elevation'] = elevations
print(f"Extracted elevations for {len(gdf[gdf['elevation'].notna()])} points.")

# 4. Transform x and y from EPSG:6514 to EPSG:4326
print("Transforming coordinates to EPSG:4326...")
try:
    gdf_wgs84 = gdf.to_crs(epsg=4326)
except Exception as e:
    print(f"Error transforming CRS: {e}")
    exit()
print(f"Transformed CRS: {gdf_wgs84.crs}")

# 5. Fill columns: name, longitude, latitude, elevation
print("Extracting longitude and latitude...")
gdf_wgs84['longitude'] = gdf_wgs84.geometry.x
gdf_wgs84['latitude'] = gdf_wgs84.geometry.y

# 6. Create final DataFrame and save to CSV
output_df = gdf_wgs84[['name', 'longitude', 'latitude', 'elevation']]

print("Sorting data by name...")
output_df = output_df.sort_values(by='name').reset_index(drop=True)

print(f"Saving data to {output_csv_path}...")
try:
    output_df.to_csv(output_csv_path, index=False, float_format='%.8f')
    print("Successfully saved CSV file.")
except Exception as e:
    print(f"Error saving CSV: {e}")

print("Script finished.")