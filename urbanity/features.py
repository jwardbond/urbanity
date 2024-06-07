import argparse
import warnings
from pathlib import Path, PurePath

import tqdm
import shapely
import osmnx as ox
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt


def area_no_buildings(
    segments: gpd.GeoDataFrame | PurePath,
    buildings: gpd.GeoDataFrame | PurePath,
    save=False,
):
    # Parse inputs
    if isinstance(segments, PurePath):
        segments = gpd.read_file(segments)
    elif not isinstance(segments, gpd.GeoDataFrame):
        raise TypeError(
            f"Expected geodataframe or pathlib path to geojson, got {type(segments)}"
        )
    segments.set_crs("EPSG:4326", allow_override=False)

    if isinstance(buildings, PurePath):
        buildings = gpd.read_file(buildings)
    elif not isinstance(buildings, gpd.GeoDataFrame):
        raise TypeError(
            f"Expected geodataframe or pathlib path to geojson, got {type(buildings)}"
        )
    buildings.set_crs("EPSG:4326", allow_override=False)

    print(segments.shape)
    print(buildings.shape)
    print(segments.head(3))

    segments = segments.overlay(buildings, how="difference")

    print(segments.shape)
    print(buildings.shape)
    print(segments.head(3))

    colors = [tuple(np.random.uniform(0, 1, 3)) for _ in segments["geometry"]]
    segments.plot(color=colors)
    plt.show()

    # for i, segment in tqdm.tqdm(segments.iterrows()):
    #     # clip all buildings, leaving only those within the segment
    #     clipped = buildings.clip(segment["geometry"])

    #     # subtract all remaining building polygons from the segment geometry

    # clipped = buildings.clip(segments.iloc[0]["geometry"])

    return segments


if __name__ == "__main__":
    filepath = Path("data/east_york_ontario/")

    area_no_buildings(
        filepath / (filepath.stem + "_segments.geojson"),
        filepath / (filepath.stem + "_osm_buildings.geojson"),
    )
