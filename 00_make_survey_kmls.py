import json
import os
from collections import defaultdict
import pyproj
from shapely.geometry import MultiPoint, Point
from shapely.ops import transform as shapely_transform

# KML template for a simple polygon
KML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <Placemark>
    <Polygon>
      <outerBoundaryIs>
        <LinearRing>
          <coordinates>
            {coordinates}
          </coordinates>
        </LinearRing>
      </outerBoundaryIs>
    </Polygon>
  </Placemark>
</Document>
</kml>
"""

# Define transformers
PROJ_NAD83_2011_UTM11N = "EPSG:6514"  # NAD83(2011) / UTM zone 11N
PROJ_WGS84 = "EPSG:4326"  # WGS84 Geographic
pyproj_transformer = pyproj.Transformer.from_crs(PROJ_NAD83_2011_UTM11N, PROJ_WGS84, always_xy=True)

def transform_shapely_geom(geom):
    """Transforms a shapely geometry from EPSG:6514 to EPSG:4326."""
    return shapely_transform(pyproj_transformer.transform, geom)

def create_kml_coordinates(polygon_exterior_coords):
    """Formats shapely polygon exterior coordinates for KML: lon,lat,alt (altitude is 0)."""
    coords_str = []
    for x, y in polygon_exterior_coords:
        coords_str.append(f"{x},{y},0")
    return " ".join(coords_str)

def main():
    geojson_path = os.path.join("data", "vector", "gcps.geojson")
    output_dir = os.path.join("planning", "flight_polygons")
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    buffer_distance = 15  # meters

    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: GeoJSON file not found at {geojson_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode GeoJSON file at {geojson_path}")
        return

    sites_data = defaultdict(list)
    for feature in data.get("features", []):
        properties = feature.get("properties", {})
        site_name = properties.get("site")
        position = properties.get("position")
        geometry = feature.get("geometry", {})
        
        if site_name and geometry.get("type") == "Point" and position != 'center':
            coordinates = geometry.get("coordinates")
            if coordinates and len(coordinates) == 2:
                sites_data[site_name].append(Point(coordinates[0], coordinates[1]))

    if not sites_data:
        print("No suitable site data found in the GeoJSON file after filtering.")
        return

    for site_name, points in sites_data.items():
        if len(points) < 3:
            print(f"Site '{site_name}' has fewer than 3 points (after filtering 'center' points), cannot compute convex hull. Skipping.")
            continue

        multi_point = MultiPoint(points)

        try:
            convex_hull_polygon = multi_point.convex_hull
        except Exception as e:
            print(f"Error calculating convex hull for site '{site_name}': {e}. Skipping.")
            continue
            
        if convex_hull_polygon.geom_type != 'Polygon':
            print(f"Convex hull for site '{site_name}' is not a polygon (likely collinear points: {convex_hull_polygon.geom_type}). Skipping.")
            continue

        buffered_polygon_utm = convex_hull_polygon.buffer(buffer_distance, cap_style=3, join_style=2)
        
        if not buffered_polygon_utm.is_valid or buffered_polygon_utm.is_empty:
            print(f"Buffering for site '{site_name}' resulted in an invalid or empty geometry. Skipping.")
            continue

        try:
            buffered_polygon_wgs84 = transform_shapely_geom(buffered_polygon_utm)
        except Exception as e:
            print(f"Error transforming coordinates for site '{site_name}': {e}. Skipping.")
            continue

        if buffered_polygon_wgs84.geom_type == 'Polygon':
            kml_coords_str = create_kml_coordinates(list(buffered_polygon_wgs84.exterior.coords))
        elif buffered_polygon_wgs84.geom_type == 'MultiPolygon':
            largest_poly = max(buffered_polygon_wgs84.geoms, key=lambda p: p.area)
            kml_coords_str = create_kml_coordinates(list(largest_poly.exterior.coords))
            print(f"Site '{site_name}' resulted in a MultiPolygon after buffering. Using the largest polygon.")
        else:
            print(f"Buffered geometry for site '{site_name}' is not a Polygon or MultiPolygon ({buffered_polygon_wgs84.geom_type}). Skipping.")
            continue

        kml_content = KML_TEMPLATE.format(coordinates=kml_coords_str)

        kml_filename = f"{site_name}.kml"
        kml_output_path = os.path.join(output_dir, kml_filename)

        try:
            with open(kml_output_path, 'w') as f:
                f.write(kml_content)
            print(f"Saved KML for site '{site_name}' to {kml_output_path}")
        except IOError:
            print(f"Error: Could not write KML file to {kml_output_path}")

if __name__ == "__main__":
    main() 