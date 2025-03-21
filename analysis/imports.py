import json
import os
from dateutil import parser
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import osmnx as ox
import geopandas as gpd

from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.affinity import rotate

AQI_DATA_FOLDER_PATH = "aqi_data/"
TRAFFIC_DATA_FOLDER_PATH = "traffic_data/"
BUS_STOP_LOCATIONS_PATH = "scarping_scripts/scarped_data/processed_data/bus_stop_locations.csv"
BUS_STOP_ROUTE_LIST_PATH = "../scarping_scripts/scarped_data/processed_data/bus_routes.txt"
