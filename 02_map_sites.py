import os
import geopandas as gpd
import rasterio
import rasterio.plot
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon
from shapely.affinity import translate
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from xml.etree import ElementTree as ET

# --- Parameters ---
imagery_url = 'https://storage.googleapis.com/mpg-aerial-survey/surveys/fixed_wing/cogs/2024_4band.tif'
imagery_epsg = 6514
gcp_path = 'data/vector/gcps.geojson'
gcp_epsg = 6514
flight_path = 'planning/flight_polygons/{site_name}.kml'
flight_epsg = 4326
flight_names = ['entrance', 'gun_range', 'indian_ridge', 'north_woodchuck', 'whaley']
output_dir = 'planning/maps'
os.makedirs(output_dir, exist_ok=True)

# --- Helper Functions ---
def parse_kml_polygon(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    coords_text = root.find('.//kml:coordinates', ns).text.strip()
    coords = [tuple(map(float, c.split(',')[:2])) for c in coords_text.split()]
    return coords

def get_overview_index(src, target_res):
    # Find the overview with resolution just coarser than target_res
    ovr_res = [src.res[0] * (2 ** i) for i in range(len(src.overviews(1)) + 1)]
    idx = np.searchsorted(ovr_res, target_res, side='right') - 1
    return max(0, min(idx, len(ovr_res) - 1))

def plot_north_arrow(ax, x, y, size=30, label='N'):
    ax.annotate('', xy=(x, y+size), xytext=(x, y),
                arrowprops=dict(facecolor='black', width=3, headwidth=12),
                ha='center', va='center')
    ax.text(x, y+size+10, label, ha='center', va='bottom', fontsize=10, fontweight='bold')

def plot_grid(ax, bounds, step=100, color='lightgray', lw=0.5):
    xmin, ymin, xmax, ymax = bounds
    for x in np.arange(np.floor(xmin/step)*step, xmax, step):
        ax.axvline(x, color=color, lw=lw, zorder=0)
    for y in np.arange(np.floor(ymin/step)*step, ymax, step):
        ax.axhline(y, color=color, lw=lw, zorder=0)

# --- Load Data ---
print('Loading GCPs...')
gcps = gpd.read_file(gcp_path)
if gcps.crs.to_epsg() != gcp_epsg:
    gcps = gcps.to_crs(epsg=gcp_epsg)

print('Loading flight polygons...')
site_polys = {}
for name in flight_names:
    coords = parse_kml_polygon(flight_path.format(site_name=name))
    points = gpd.points_from_xy([c[0] for c in coords], [c[1] for c in coords])
    poly_geom = Polygon(points)
    poly = gpd.GeoSeries([poly_geom], crs=f'EPSG:{flight_epsg}')
    poly = poly.to_crs(epsg=gcp_epsg)
    site_polys[name] = poly.iloc[0]
all_flights = gpd.GeoDataFrame({'site': flight_names, 'geometry': [site_polys[n] for n in flight_names]}, crs=f'EPSG:{gcp_epsg}')

# --- Load Imagery ---
print('Opening imagery...')
with rasterio.open(imagery_url) as src:
    print(f"Source CRS: {src.crs}")
    print(f"Source shape: {src.shape}")
    
    # --- All Sites Map ---
    print('Reading imagery for all sites overview...')
    all_bounds = all_flights.total_bounds
    
    # Calculate window for all sites with small buffer
    buffer = 500
    buffered_bounds = [all_bounds[0] - buffer, all_bounds[1] - buffer, 
                      all_bounds[2] + buffer, all_bounds[3] + buffer]
    
    # Get window from bounds
    all_window = rasterio.windows.from_bounds(*buffered_bounds, src.transform)
    
    # Calculate thumbnail size for overview (aim for reasonable size)
    thumb_width = 800
    thumb_height = int(all_window.height * (thumb_width / all_window.width))
    
    # Read RGB bands within the window, downsampled
    all_img = src.read([1, 2, 3],
                      window=all_window,
                      out_shape=(3, thumb_height, thumb_width),
                      resampling=rasterio.enums.Resampling.bilinear)
    
    print(f"All sites image shape: {all_img.shape}")
    
    # Normalize if necessary (assuming values might be > 1)
    if all_img.max() > 1.0:
        all_img = all_img / 255.0 if all_img.max() > 1 else all_img
    
    # Transpose for imshow (bands last): (3, H, W) -> (H, W, 3)
    all_img_display = np.transpose(all_img, (1, 2, 0))
    print(f"All sites display shape: {all_img_display.shape}")
    
    # Get the precise bounds of the window we read
    all_window_bounds = rasterio.windows.bounds(all_window, src.transform)
    all_extent = [all_window_bounds[0], all_window_bounds[2], all_window_bounds[1], all_window_bounds[3]]
    
    # --- Individual Site Images ---
    print('Reading imagery for individual sites...')
    site_imgs = {}
    
    for name, poly in site_polys.items():
        print(f"Processing site: {name}")
        
        # Buffer the polygon
        buf_poly = poly.buffer(30)
        bounds = buf_poly.bounds
        
        # Get window from bounds
        window = rasterio.windows.from_bounds(*bounds, src.transform)
        
        # Calculate thumbnail size for site (higher resolution than overview)
        site_thumb_width = 600
        site_thumb_height = int(window.height * (site_thumb_width / window.width))
        
        # Read RGB bands within the window, downsampled
        img = src.read([1, 2, 3],
                      window=window,
                      out_shape=(3, site_thumb_height, site_thumb_width),
                      resampling=rasterio.enums.Resampling.bilinear)
        
        # Normalize if necessary
        if img.max() > 1.0:
            img = img / 255.0 if img.max() > 1 else img
        
        # Transpose for imshow (bands last)
        img_display = np.transpose(img, (1, 2, 0))
        
        # Get the precise bounds of the window we read
        window_bounds = rasterio.windows.bounds(window, src.transform)
        extent = [window_bounds[0], window_bounds[2], window_bounds[1], window_bounds[3]]
        
        site_imgs[name] = (img_display, extent, bounds)

# --- Plot All Sites Map --- 
print('Plotting all sites map...')
fig, ax = plt.subplots(figsize=(10, 10))

# Plot imagery using imshow with extent (like your working example)
ax.imshow(all_img_display, extent=all_extent, zorder=0)

# Plot grid
plot_grid(ax, buffered_bounds, step=100)

# Plot GCPs: white dot behind cyan dot
ax.scatter(gcps.geometry.x, gcps.geometry.y, c='white', s=60, label='_nolegend_', zorder=3, edgecolor='none')
ax.scatter(gcps.geometry.x, gcps.geometry.y, c='cyan', s=30, label='GCP', zorder=4, edgecolor='black', linewidth=0.5)

# Plot flight polygons
for name, poly in site_polys.items():
    x, y = poly.exterior.xy
    ax.plot(x, y, color='blue', lw=2, label='Flight bounds' if name==flight_names[0] else None, zorder=5)

# Add site labels north of each poly
for name, poly in site_polys.items():
    centroid = poly.centroid
    north_pt = translate(centroid, yoff=60)
    ax.text(north_pt.x, north_pt.y, name.replace('_', '\n'), ha='center', va='bottom', 
           fontsize=11, fontweight='bold', color='black', 
           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

# Legend (cyan for GCP)
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markeredgecolor='black', markersize=8, label='GCP'),
           Line2D([0], [0], color='blue', lw=2, label='Flight bounds')]
ax.legend(handles=handles, loc='lower left')

ax.set_title('All Sites Overview')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.set_xlim(buffered_bounds[0], buffered_bounds[2])
ax.set_ylim(buffered_bounds[1], buffered_bounds[3])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'all_sites.png'), dpi=300)
plt.close()

# --- Plot Each Site ---
print('Plotting individual site maps...')
for name in flight_names:
    img_display, extent, bounds = site_imgs[name]
    poly = site_polys[name]
    buf_poly = poly.buffer(30)
    
    # Filter GCPs for this site (with buffer)
    gcp_mask = gcps.within(buf_poly)
    site_gcps = gcps[gcp_mask]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot imagery using imshow with extent
    ax.imshow(img_display, extent=extent, zorder=0)
    
    plot_grid(ax, bounds, step=20)
    
    # GCPs: white dot behind cyan dot
    ax.scatter(site_gcps.geometry.x, site_gcps.geometry.y, c='white', s=60, label='_nolegend_', zorder=3, edgecolor='none')
    ax.scatter(site_gcps.geometry.x, site_gcps.geometry.y, c='cyan', s=30, label='GCP', zorder=4, edgecolor='black', linewidth=0.5)
    
    # Flight poly
    x, y = poly.exterior.xy
    ax.plot(x, y, color='blue', lw=2, label='Flight bounds', zorder=5)
    
    # Legend (keep position as before: lower right)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markeredgecolor='black', markersize=8, label='GCP'),
               Line2D([0], [0], color='blue', lw=2, label='Flight bounds')]
    ax.legend(handles=handles, loc='lower right')
    
    ax.set_title(f'{name.replace("_", " ").title()} Site')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}_site.png'), dpi=300)
    plt.close()

print('Done! Maps saved to', output_dir)