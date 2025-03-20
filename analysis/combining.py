import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.colors as mcolors
from dateutil import parser

bus_stop_location_filepath = "scarping_scripts/scarped_data/processed_data/bus_stop_locations.csv"
aqi_data_filepath = "aqi_data/aqi_data_Sector_22.xlsx"

def haversine(lon1, lat1, lon2, lat2):
    # --- Helper function: Haversine distance ---
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # Earth radius in kilometers
    return km

def get_bus_stops_within_buffer(center_point, buffer_radius_km):
    # --- Load and filter bus stop data ---
    bus_stops = pd.read_csv(bus_stop_location_filepath)
    # Note: haversine expects (lon, lat) for the center point.
    bus_stops["distance_km"] = bus_stops.apply(
        lambda row: haversine(center_point[1], center_point[0], row["long"], row["lat"]),
        axis=1
    )
    bus_stops_within_buffer = bus_stops[
        (bus_stops["distance_km"] <= buffer_radius_km) & 
        (~bus_stops["name"].str.contains("school", case=False, na=False)) &
        (~bus_stops["name"].str.contains("guru", case=False, na=False)) &
        (~bus_stops["name"].str.contains("community", case=False, na=False)) &
        (~bus_stops["name"].str.contains("park", case=False, na=False))
    ]

    return bus_stops_within_buffer

def get_sectors_in_buffer(buffer_polygon):
    # --- Get sector boundaries from OSM ---
    # Query for administrative boundaries (admin_level=10) and filter for those with "Sector" in the name

    tags = {'boundary': 'administrative', 'admin_level': '10'}
    sectors_gdf = ox.features_from_place("Chandigarh, India", tags=tags)
    sectors_gdf = sectors_gdf[sectors_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    sectors_gdf = sectors_gdf[sectors_gdf['name'].str.contains('Sector', case=False, na=False)]
    # Keep only sectors that intersect our buffer
    return sectors_gdf[sectors_gdf.intersects(buffer_polygon)]

def get_graph_and_buffer(center_point, buffer_radius_km):
    # --- Get the street network graph (1 km buffer) ---
    graph = ox.graph_from_point(center_point, dist=buffer_radius_km * 1000, network_type="all")

    # --- Create a circular buffer polygon around the center ---
    center_geom = Point(center_point[1], center_point[0])
    center_gdf = gpd.GeoDataFrame({'geometry': [center_geom]}, crs="EPSG:4326")
    # Project to a metric CRS (UTM zone for Chandigarh, e.g., EPSG:32643)
    center_gdf_proj = center_gdf.to_crs(epsg=32643)
    buffer_polygon_proj = center_gdf_proj.buffer(buffer_radius_km * 1000).iloc[0]  # 1000 m = 1 km
    # Reproject back to lat/lon (EPSG:4326)
    buffer_polygon = gpd.GeoSeries([buffer_polygon_proj], crs=center_gdf_proj.crs).to_crs(epsg=4326).iloc[0]

    return graph, buffer_polygon

def plot_graph_and_bus_stops(ax, graph, sector_labels=True, bus_stops=True, show_center_point=True, sector_boundaries=False):
    # Plot the street network graph
    ox.plot_graph(graph, ax=ax, node_size=0, edge_color="gray", edge_linewidth=0.5, show=False)

    
    # Plot sector boundaries with different colors and labels
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, (idx, sector) in enumerate(sectors_in_buffer.iterrows()):
        color = colors[i % len(colors)]
        sector_boundaries_aplha = 0.7 if sector_boundaries else 0
        gpd.GeoSeries(sector.geometry).plot(ax=ax, facecolor='none', edgecolor=color, linewidth=2, alpha=sector_boundaries_aplha,
                                            label=sector.get('name', f"Sector {i}"))
        
        centroid = sector.geometry.centroid
        if sector_labels:
            ax.text(centroid.x, centroid.y, sector.get('name', f"Sector {i}"), fontsize=9, color=color)

    if show_center_point:
        # Plot the center point (AQI Station)
        ax.scatter(center_point[1], center_point[0], s=100, color="red", zorder=5, label="AQI Station")

    if bus_stops:
        # Plot bus stops within the buffer
        for _, row in bus_stops_within_buffer.iterrows():
            ax.scatter(row["long"], row["lat"], s=50, color="blue", zorder=6)
            ax.text(row["long"], row["lat"], row["name"], fontsize=9, zorder=7)

def get_closest_AQI_data(timestamp_input):
    df = pd.read_excel(aqi_data_filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Parse the input string to get a datetime, then extract the time-of-day in seconds
    input_dt = parser.parse(timestamp_input)
    target_seconds = input_dt.hour * 3600 + input_dt.minute * 60 + input_dt.second
    
    # Compute seconds since midnight for each row in the Timestamp column
    df['seconds'] = df['Timestamp'].dt.hour * 3600 + df['Timestamp'].dt.minute * 60 + df['Timestamp'].dt.second
    
    # Compute absolute difference in seconds, adjusting for circularity of the clock
    # (if the difference is > 12 hours, take the complement relative to 24h)
    df['diff'] = (df['seconds'] - target_seconds).abs()
    df['diff'] = df['diff'].apply(lambda x: x if x <= 43200 else 86400 - x)
    
    # Find and return the row with the smallest difference
    closest_idx = df['diff'].idxmin()
    return df.loc[closest_idx]
    

# --- Define center point and buffer (Sector 22 / AQI Station) ---
center_point = (30.735567, 76.775714)  # (lat, lon)
buffer_radius_km = 1.0

bus_stops_within_buffer = get_bus_stops_within_buffer(center_point, buffer_radius_km)

graph, buffer_polygon = get_graph_and_buffer(center_point, buffer_radius_km)

sectors_in_buffer = get_sectors_in_buffer(buffer_polygon=buffer_polygon)

# --- Load and plot traffic data ---
traffic_data_files = [
    # "traffic_data/2025-03-09_16-41-08.json", "9th March 11:09 am",
    # "traffic_data/2025-03-14_22-00-06.json", "14th March 4:30 pm",
    "traffic_data/2025-03-14_23-07-25.json", "14th March 5:30 pm",
    "traffic_data/2025-03-19_13-28-11.json", "19th March 7:55 am"
    # "traffic_data/2025-03-19_16-53-43.json", "19th March 11:21 am"
]

metrics = ["jamFactor", "freeFlow", "speed"]
metrics = ["speedUncapped"]
aqi_data_per_timestamp = []
sampling_time_list = []

for metric in metrics:
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))  # Create a 1x2 subplot

    max_metric_values_per_time = []
    for i in range(0, len(traffic_data_files), 2):
        filename = traffic_data_files[i]
        with open(filename, "r") as f:
            traffic_data = json.load(f)

        metric_values_per_time = [location['currentFlow'][metric] for location in traffic_data['results']]
        max_metric_values_per_time.append(max(metric_values_per_time))

    for i in range(0, len(traffic_data_files), 2):
        ax = axs[i // 2]  # Select correct subplot

        filename = traffic_data_files[i]
        
        sampling_time = traffic_data_files[i + 1]
        if sampling_time not in sampling_time_list:
            sampling_time_list.append(sampling_time)

        closed_aqi_data = get_closest_AQI_data(sampling_time)
        existing_timestamps = [entry["Timestamp"] for entry in aqi_data_per_timestamp]
        if closed_aqi_data["Timestamp"] not in existing_timestamps:
            aqi_data_per_timestamp.append(closed_aqi_data)

        with open(filename, "r") as f:
            traffic_data = json.load(f)

        max_metric = max(max_metric_values_per_time)

        plot_graph_and_bus_stops(ax=ax, graph=graph, sector_labels=False, bus_stops=True, show_center_point=False)

        cmap = plt.get_cmap('GnBu')
        norm = mcolors.Normalize(vmin=0, vmax=max_metric)
        # print(f"Maximum {metric} in {filename} is {max_metric}")

        all_metric_values_for_timestamp = []

        for location in traffic_data['results']:
            metric_value = location['currentFlow'][metric]
            all_metric_values_for_timestamp.append(metric_value)
            color = cmap(norm(metric_value))
            links = location['location']['shape']['links']
            for link in links:
                pts = link['points']
                if any(buffer_polygon.contains(Point(pt['lng'], pt['lat'])) for pt in pts):
                    xs = [pt['lng'] for pt in pts]
                    ys = [pt['lat'] for pt in pts]
                    ax.plot(xs, ys, color=color, linewidth=2, zorder=7)        

        # mean, var, std for timestamp
        all_metric_values_for_timestamp = np.array(all_metric_values_for_timestamp)
        print(f"{metric} mean for time index {i} = {all_metric_values_for_timestamp.mean()}")
        print(f"{metric} var for time index {i} = {all_metric_values_for_timestamp.var()}")
        print(f"{metric} std for time index {i} = {all_metric_values_for_timestamp.std()}\n")

        ax.set_title(f"{metric} - {sampling_time}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label(metric)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

columns = ["AQI", "Dominant Pollutant", "CO", "NO2", "O3", "PM10", "PM2.5", "SO2", 
           "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Pressure (hPa)"]

aqi_metrics_per_timestamp = [
    [aqi_data_per_timestamp[0][col] for col in columns],
    [aqi_data_per_timestamp[1][col] for col in columns]
]

# Create figure
fig, ax = plt.subplots(figsize=(12.5, 2))
ax.set_frame_on(False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

print([data["Timestamp"] for data in aqi_data_per_timestamp])

# Create table
table = plt.table(cellText=aqi_metrics_per_timestamp,
                  colLabels=columns,
                  rowLabels=[data["Timestamp"] for data in aqi_data_per_timestamp],
                  cellLoc='center',
                  loc='center')

table.auto_set_font_size(True)
# table.set_fontsize(10)
table.auto_set_column_width([i for i in range(len(columns))])
plt.title("AQI Data Comparison")

plt.show()
