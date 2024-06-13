import sys
from pathlib import Path, PurePath

import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt


def _input_to_geodf(
    x: gpd.geodataframe.GeoDataFrame | gpd.geoseries.GeoSeries | PurePath,
):
    """Util function used to parse function inputs."""
    if isinstance(x, PurePath):
        x = gpd.read_file(x)
        x.set_crs("EPSG:4326", allow_override=False)
    elif not (
        isinstance(x, gpd.geodataframe.GeoDataFrame)
        or isinstance(x, gpd.geoseries.GeoSeries)
    ):
        raise TypeError(
            f"Expected geodataframe or pathlib path to geojson, got {type(x)}"
        )

    return x


def _save_geodf_with_prompt(x: gpd.GeoDataFrame, savepath: PurePath):
    if savepath.exists():
        prompt_success = False
        while not prompt_success:
            overwrite = str(input(f"Overwriting {savepath}. Proceed? (Y/N)"))
            if overwrite == "Y" or overwrite == "y":
                prompt_success = True
                savepath.write_text(x.to_json())
            elif overwrite == "N" or overwrite == "n":
                prompt_success = True
                sys.exit("Exiting")


def subtract_polygons(
    segments: gpd.GeoDataFrame | PurePath,
    polygons: gpd.GeoDataFrame | PurePath,
    savepath=None,
):
    """Given segments and a list of polygons representing areas to NOT include, add various metrics

    This is basically a wrapper for geopandas overlay diff. Works on a copy of inputs.

    Args:
        segments (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of segments or path to a .geojson containing them.
        polygons (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of polygons to subtract, or path to a .geojson containing them.
        savepath (pathlib.PurePath, optional): Folder to save result to. Defaults to None (not saved)

    Returns:
        geopandas.GeoDataframe: A geodataframe containing the set difference segments - polygons
    """

    # Parse inputs
    segments = segments.copy()
    polygons = polygons.copy()

    segments = _input_to_geodf(segments)
    polygons = _input_to_geodf(polygons)

    # Get set difference
    segments = segments.overlay(polygons, how="difference")

    # Save / overwrite segment file
    if savepath:
        savepath = savepath / (savepath.stem + "_segments.geojson")
        _save_geodf_with_prompt(segments, savepath)

    return segments


def add_building_features(
    segments: gpd.GeoDataFrame | PurePath,
    buildings: gpd.GeoDataFrame | PurePath,
    savepath=None,
):
    """Given segments and a list of polygons representing areas to NOT include, add various metrics

    Segments and polygons should be in the same crs.

    Args:
        segments (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of segments or path to a .geojson containing them.
        buildings (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of polygons to subtract, or path to a .geojson containing them.
        savepath (pathlib.PurePath, optional): Folder to save result to. Defaults to None (not saved)

    Returns:
        geopandas.GeoDataframe: A geodataframe containing the set difference segments - polygons
    """

    # Parse inputs
    segments = segments.copy()
    buildings = buildings.copy()

    segments = _input_to_geodf(segments)
    buildings = _input_to_geodf(buildings)

    # Convert to projected crs for better area calculations
    segments = segments.to_crs("EPSG:3587")
    buildings = buildings.to_crs("EPSG:3587")

    # Get set difference
    diff = segments.overlay(buildings, how="difference")

    # Make sure it is indexed correctly
    diff = diff.set_index(diff["id"].astype(str).astype(int), drop=False)
    diff.index.names = [None]
    segments = segments.set_index(segments["id"].astype(str).astype(int), drop=False)
    segments.index.names = [None]

    # Calculating areas
    segments["area"] = segments["geometry"].area
    segments["unused_area"] = diff["geometry"].area
    segments["area_ratio"] = segments["unused_area"] / segments["area"]

    # TODO plotting, take out
    # diff["area_ratio"] = segments["area_ratio"]
    # diff.plot(column="area_ratio")
    # plt.show()

    # Save / overwrite segment file
    if savepath:
        savepath = savepath / (savepath.stem + "_segments.geojson")
        _save_geodf_with_prompt(segments, savepath)

    return segments


if __name__ == "__main__":
    filepath = Path("data/east_york_ontario/")

    add_building_features(
        filepath / (filepath.stem + "_segments.geojson"),
        filepath / (filepath.stem + "_osm_buildings_parks.geojson"),
    )
