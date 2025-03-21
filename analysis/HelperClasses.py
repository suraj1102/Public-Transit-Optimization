from imports import *

class GeoUtils:
    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        """Compute the great-circle distance between two points."""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c  # Earth radius in kilometers
        return km

    @staticmethod
    def get_bus_stops_within_buffer(center_point, buffer_radius_km, bus_stop_location_filepath=BUS_STOP_LOCATIONS_PATH):
        """Load bus stop data and return only stops within a given buffer around the center point."""
        bus_stops = pd.read_csv(bus_stop_location_filepath)
        bus_stops["distance_km"] = bus_stops.apply(
            lambda row: GeoUtils.haversine(center_point[1], center_point[0], row["long"], row["lat"]),
            axis=1
        )
        return bus_stops[bus_stops["distance_km"] <= buffer_radius_km]

    @staticmethod
    def get_graph_and_buffer(center_point, buffer_radius_km):
        """Retrieve the street network graph and compute a circular buffer around the center."""
        graph = ox.graph_from_point(center_point, dist=buffer_radius_km * 1000, network_type="all")
        center_geom = Point(center_point[1], center_point[0])
        center_gdf = gpd.GeoDataFrame({'geometry': [center_geom]}, crs="EPSG:4326")
        center_gdf_proj = center_gdf.to_crs(epsg=32643)
        buffer_polygon_proj = center_gdf_proj.buffer(buffer_radius_km * 1000).iloc[0]
        buffer_polygon = gpd.GeoSeries([buffer_polygon_proj], crs=center_gdf_proj.crs).to_crs(epsg=4326).iloc[0]
        return graph, buffer_polygon

    @staticmethod
    def get_sectors_in_buffer(buffer_polygon):
        """Query OSM for sector boundaries in Chandigarh and filter those intersecting the buffer."""
        tags = {'boundary': 'administrative', 'admin_level': '10'}
        sectors_gdf = ox.features_from_place("Chandigarh, India", tags=tags)
        sectors_gdf = sectors_gdf[sectors_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        sectors_gdf = sectors_gdf[sectors_gdf['name'].str.contains('Sector', case=False, na=False)]
        return sectors_gdf[sectors_gdf.intersects(buffer_polygon)]


class DataLoader:
    def __init__(self, aqi_path, traffic_path, bus_stop_loc_path):
        self.aqi_path = aqi_path
        self.traffic_path = traffic_path
        self.bus_stop_loc_path = bus_stop_loc_path

        self.aqi_df = None
        self.traffic_data = None
        self.bus_stop_locations_df = None
        
        self.traffic_timestamp = None
        self.aqi_location_name = None

        self.load_traffic_data()
        self.load_aqi_data()
        self.load_bus_stops()

    # Load In Data:
    def load_traffic_data(self):
        """Load traffic data from a JSON file."""
        with open(self.traffic_path, "r") as f:
            self.traffic_data = json.load(f)
            return self.traffic_data

    def load_bus_stops(self):
        """Load bus stop data from a CSV file."""
        self.bus_stop_locations_df = pd.read_csv(self.bus_stop_loc_path)
        return self.bus_stop_locations_df
    
    def load_aqi_data(self):
        """Load AQI data from an Excel file."""
        df = pd.read_excel(self.aqi_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        self.aqi_df = df
        self.aqi_location_name = df["Location"].iloc[0]
        return self.aqi_df

    # Helper Functions 
    def get_traffic_timestamp(self):
        """Get timestamp from traffic data file and convert it to readable 
        format (also the format which goes into the get_closest_AQI_data function)"""
        if not self.traffic_data:
            return None

        timestamp = self.traffic_data["sourceUpdated"]
        dt = parser.isoparse(timestamp)
        
        # Use a format that's consistent across platforms
        day = str(dt.day)  # Remove leading zeros this way
        formatted_date = f"{day} {dt.strftime('%B %I:%M %p')}"
        
        return formatted_date

    def get_closest_AQI_data(self, timestamp_input):
        """Return the row from the AQI dataframe whose time-of-day is closest to the input."""
        if self.aqi_df is None or self.aqi_df.empty:
            return None
        
        df = self.aqi_df
        timestamp_input = timestamp_input.strip()
        
        # Use the more flexible parser instead of strptime
        try:
            input_dt = parser.parse(timestamp_input)
        except ValueError:
            print(f"Error parsing timestamp: {timestamp_input}")
            return None
        
        target_seconds = input_dt.hour * 3600 + input_dt.minute * 60 + input_dt.second
        df['seconds'] = df['Timestamp'].dt.hour * 3600 + df['Timestamp'].dt.minute * 60 + df['Timestamp'].dt.second
        df['diff'] = (df['seconds'] - target_seconds).abs()
        df['diff'] = df['diff'].apply(lambda x: x if x <= 43200 else 86400 - x)
        closest_idx = df['diff'].idxmin()
        return df.loc[closest_idx]

    def get_aqi_station_location(self):
        """Returns (lat, long) of aqi center"""
        if self.aqi_df is None or self.aqi_df.empty:
            return None
        
        return self.aqi_df["Latitude"].iloc[0], self.aqi_df["Longitude"].iloc[0]

    def get_metric_values_in_buffer(self, metric, buffer):
        metric_values_list = []
        for location in self.traffic_data['results']:
            metric_value = location['currentFlow'][metric]
            links = location['location']['shape']['links']
            for link in links:
                pts = link['points']
                if any(buffer.contains(Point(pt['lng'], pt['lat'])) for pt in pts):
                    metric_values_list.append(metric_value)
        return metric_values_list


class Visualize:
    def __init__(self, dataLoader:DataLoader, buffer_radius_km):
        self.dataLoader = dataLoader
        self.center_point = dataLoader.get_aqi_station_location()
        self.buffer_radius_km = buffer_radius_km

        # Get Graph and Sector Labels within buffer radius
        self.graph, self.buffer_polygon = GeoUtils.get_graph_and_buffer(self.center_point, buffer_radius_km)
        self.sectors_in_graph_gdf = GeoUtils.get_sectors_in_buffer(self.buffer_polygon)

        # Get Bus Stops within buffer radius
        self.bus_stops_in_buffer = GeoUtils.get_bus_stops_within_buffer(self.center_point, buffer_radius_km)      

    def plot_graph(self, ax, show_center_pt=False):
        ox.plot_graph(self.graph, ax=ax, node_size=0, edge_color="gray", edge_linewidth=0.5, show=False)
        if show_center_pt:
            ax.scatter(self.center_point[1], self.center_point[0], s=100, color="red", zorder=5, label="AQI Station")

    def plot_sector_info(self, ax, show_border=False):
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, (idx, sector) in enumerate(self.sectors_in_graph_gdf.iterrows()):
            color = colors[i % len(colors)]
            sector_boundaries_aplha = 0.7 if show_border else 0
            gpd.GeoSeries(sector.geometry).plot(ax=ax, facecolor='none', edgecolor=color, linewidth=2, alpha=sector_boundaries_aplha,
                                                label=sector.get('name', f"Sector {i}"))
            
            centroid = sector.geometry.centroid
            ax.text(centroid.x, centroid.y, sector.get('name', f"Sector {i}"), fontsize=9, color=color)

    def plot_bus_stops(self, ax):
        for _, row in self.bus_stops_in_buffer.iterrows():
            ax.scatter(row["long"], row["lat"], s=50, color="blue", zorder=6, alpha=0.6)
            ax.text(row["long"], row["lat"], row["name"], fontsize=9, zorder=7)

    def plot_metric(self, ax, metric, norm_between=(None,None), cmap_name=None):
        # Find max metric value to scale cmap
        if not norm_between[0] or not norm_between[1] or not cmap_name:
            metric_values_list = self.dataLoader.get_metric_values_in_buffer(metric, self.buffer_polygon)

            cmap = plt.get_cmap('GnBu')
            norm = mcolors.Normalize(vmin=min(metric_values_list), vmax=max(metric_values_list)) 
        else:
            cmap = plt.get_cmap(cmap_name)
            norm = mcolors.Normalize(vmin=norm_between[0], vmax=norm_between[1])

        for location in self.dataLoader.traffic_data['results']:
            metric_value = location['currentFlow'][metric]
            links = location['location']['shape']['links']
            color = cmap(norm(metric_value))
            for link in links:
                pts = link['points']
                if any(self.buffer_polygon.contains(Point(pt['lng'], pt['lat'])) for pt in pts):
                    xs = [pt['lng'] for pt in pts]
                    ys = [pt['lat'] for pt in pts]
                    ax.plot(xs, ys, color=color, linewidth=2, zorder=1)

        return cmap, norm # for plotting cmap
    
    def plot_cmap(self, fig, cmap, norm, metric, pos=[0.92, 0.2, 0.02, 0.6]):
        cax = fig.add_axes(pos)  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cax).set_label(metric)
        plt.tight_layout(rect=[0, 0, 0.9, 1])

    def plot_aqi(self, ax, formatted_time_stamp_list):
        if isinstance(formatted_time_stamp_list, str):
            raise TypeError("Expected a list of timestamps, but received a single string. Please provide a list.")

        ax.set_frame_on(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        columns = ["Timestamp", "AQI", "Dominant Pollutant", "CO", "NO2", "O3", "PM10", "PM2.5", "SO2", 
                    "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Pressure (hPa)"]

        aqi_metrics_per_timestamp = [
            aqi_data[columns] for t in formatted_time_stamp_list
            if (aqi_data := self.dataLoader.get_closest_AQI_data(t)) is not None
        ]

        table = plt.table(cellText=aqi_metrics_per_timestamp,
                        colLabels=columns,
                        # rowLabels=[data["Timestamp"] for data in aqi_metrics_per_timestamp],
                        cellLoc='center',
                        loc='center')

        table.auto_set_font_size(True)
        table.auto_set_column_width([i for i in range(len(columns))])


#####################################
########### Usage ###################
#####################################

def show_single_traffic_timestamp_plot():
    dataLoader = DataLoader(
        AQI_DATA_FOLDER_PATH + "aqi_data_Sector_22.xlsx",
        TRAFFIC_DATA_FOLDER_PATH + "2025-03-14_22-00-06.json",
        BUS_STOP_LOCATIONS_PATH
    )

    traffic_t = dataLoader.get_traffic_timestamp()
    closed_aqi_t = dataLoader.get_closest_AQI_data(traffic_t)
    center_p = dataLoader.get_aqi_station_location()
    aqi_location_name = dataLoader.aqi_location_name
    
    viz = Visualize(dataLoader, 1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    viz.plot_graph(ax)
    cmap, norm = viz.plot_metric(ax, "speed")
    viz.plot_cmap(fig, cmap, norm, "speed")
    viz.plot_bus_stops(ax)
    viz.plot_sector_info(ax)
    ax.set_title(f"Traffic and Bus Stops in {aqi_location_name} at {traffic_t}")

    fig, ax = plt.subplots(figsize=(12, 4))
    viz.plot_aqi(ax, [traffic_t])
    ax.set_title(f"AQI at {aqi_location_name}")

    plt.show()


def compare_two_timestamps():
    dl1 = DataLoader(
        AQI_DATA_FOLDER_PATH + "aqi_data_Sector_22.xlsx",
        TRAFFIC_DATA_FOLDER_PATH + "2025-03-14_22-00-06.json",
        BUS_STOP_LOCATIONS_PATH
    )

    dl2 = DataLoader(
        AQI_DATA_FOLDER_PATH + "aqi_data_Sector_22.xlsx",
        TRAFFIC_DATA_FOLDER_PATH + "2025-03-09_16-41-08.json",
        BUS_STOP_LOCATIONS_PATH
    )

    viz1 = Visualize(dl1, 1)
    viz2 = Visualize(dl2, 1)

    t1 = dl1.get_traffic_timestamp()
    t2 = dl2.get_traffic_timestamp()

    metric = "speed"
    dl1_metric_values = dl1.get_metric_values_in_buffer(metric, viz1.buffer_polygon)
    dl2_metric_values = dl2.get_metric_values_in_buffer(metric, viz2.buffer_polygon)

    cmap_name = "winter_r"
    norm_between = (
        min(min(dl1_metric_values), min(dl2_metric_values)),
        max(max(dl1_metric_values), max(dl2_metric_values))
    )
    
    fig, axs = plt.subplots(1, 2, figsize=(15,8))
    
    ax = axs[0]
    viz1.plot_graph(ax)
    viz1.plot_metric(ax, metric, norm_between=norm_between, cmap_name=cmap_name)

    ax = axs[1]
    viz2.plot_graph(ax)
    cmap, norm = viz2.plot_metric(ax, metric, norm_between=norm_between, cmap_name=cmap_name)

    viz1.plot_cmap(fig, cmap, norm, metric)
    # can also do viz2.plot_cmap(fig, cmap, norm, metric)

    axs[0].set_title(f"{metric} - {dl1.get_traffic_timestamp()}")
    axs[1].set_title(f"{metric} - {dl2.get_traffic_timestamp()}")


    fig2, ax2 = plt.subplots(figsize=(12, 4))
    viz1.plot_aqi(ax2, [t1, t2])
    ax2.set_title("AQI Comparison")

    plt.show()


if __name__ == "__main__":
    # show_single_traffic_timestamp_plot()
    compare_two_timestamps()

