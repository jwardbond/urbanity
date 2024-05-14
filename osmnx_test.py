import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import geopandas as gpd
import osmnx as ox
import pandas as pd

from genregion import generate_regions


def get_and_plot(loc: str):
    G = ox.graph_from_place(loc, network_type="drive", simplify=False)
    M = ox.convert.to_undirected(G)

    nc = [
        "r" if ox.simplification._is_endpoint(G, node, None) else "y"
        for node in G.nodes()
    ]
    fig, ax = ox.plot_graph(M, node_color=nc)

    _, gdf = ox.convert.graph_to_gdfs(M)

    edges = gdf["geometry"].to_list()

    urban_regions = generate_regions(
        edges,
        grid_size=1024,
        area_thres=10000,
        width_thres=20,
        clust_width=25,
        point_precision=5,
    )

    polygons = gpd.GeoSeries(urban_regions)
    polygons.plot()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for osmnx")

    parser.add_argument(
        "location", type=str, help="Location string to get road network map"
    )

    args = parser.parse_args()

    get_and_plot(args.location)

    get_and_plot()
