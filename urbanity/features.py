import sys
from pathlib import Path, PurePath

import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt


# def _area_no_buildings(
#     segments: gpd.GeoDataFrame | PurePath,
#     buildings: gpd.GeoDataFrame | PurePath,
#     savepath: PurePath = None,
# ):
#     """Given building polygons, subtracts them from segment areas

#     Segments and buildings should be in the same crs.

#     Args:
#         segments (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of segments or path to a .geojson containing them.
#         buildings (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of buildings or path to a .geojson containing them.
#         savepath (pathlib.PurePath, optional): Folder to save result to. Defaults to None (not saved)

#     Returns:
#         geopandas.GeoDataframe: A geodataframe containing the set difference segments - buildings
#     """
#     # Parse inputs
#     if isinstance(segments, PurePath):
#         segments = gpd.read_file(segments)
#     elif not isinstance(segments, gpd.GeoDataFrame):
#         raise TypeError(
#             f"Expected geodataframe or pathlib path to geojson, got {type(segments)}"
#         )
#     segments.set_crs("EPSG:4326", allow_override=False)

#     if isinstance(buildings, PurePath):
#         buildings = gpd.read_file(buildings)
#     elif not isinstance(buildings, gpd.GeoDataFrame):
#         raise TypeError(
#             f"Expected geodataframe or pathlib path to geojson, got {type(buildings)}"
#         )
#     buildings.set_crs("EPSG:4326", allow_override=False)

#     print(segments.shape)
#     print(buildings.shape)
#     print(segments.head(3))

#     segments = segments.overlay(buildings, how="difference")

#     print(segments.shape)
#     print(buildings.shape)
#     print(segments.head(3))

#     # Save / overwrite segment file
#     if savepath:
#         filepath = savepath / (savepath.stem + "_segments.geojson")

#         if filepath.exists():
#             prompt_success = False
#             while not prompt_success:
#                 overwrite = str(input(f"Overwriting {filepath}. Proceed? (Y/N)"))
#                 if overwrite == "Y" or overwrite == "y":
#                     prompt_success = True
#                     filepath.write_text(segments.to_json())
#                 elif overwrite == "N" or overwrite == "n":
#                     prompt_success = True
#                     sys.exit("Exiting")

#     # colors = [tuple(np.random.uniform(0, 1, 3)) for _ in segments["geometry"]]
#     # segments.plot(color=colors)
#     # plt.show()

#     return segments


def add_osm_coverage_features(
    segments: gpd.GeoDataFrame | PurePath,
    polygons: gpd.GeoDataFrame | PurePath,
    savepath=None,
):
    """Given segments and a list of polygons representing areas to NOT include, add various metrics

    Segments and polygons should be in the same crs.

    Args:
        segments (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of segments or path to a .geojson containing them.
        polygons (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of polygons to subtract, or path to a .geojson containing them.
        savepath (pathlib.PurePath, optional): Folder to save result to. Defaults to None (not saved)

    Returns:
        geopandas.GeoDataframe: A geodataframe containing the set difference segments - polygons
    """

    # Parse inputs
    if isinstance(segments, PurePath):
        segments = gpd.read_file(segments)
        segments.set_crs("EPSG:4326", allow_override=False)
    elif not isinstance(segments, gpd.GeoDataFrame):
        raise TypeError(
            f"Expected geodataframe or pathlib path to geojson, got {type(segments)}"
        )

    if isinstance(polygons, PurePath):
        polygons = gpd.read_file(polygons)
        polygons.set_crs("EPSG:4326", allow_override=False)
    elif not isinstance(polygons, gpd.GeoDataFrame):
        raise TypeError(
            f"Expected geodataframe or pathlib path to geojson, got {type(polygons)}"
        )

    # Convert to projected crs for better area calculations
    segments = segments.to_crs("EPSG:3587")
    polygons = polygons.to_crs("EPSG:3587")

    # Get set difference
    diff = segments.overlay(polygons, how="difference")

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

    diff["area_ratio"] = segments["area_ratio"]
    diff.plot(column="area_ratio")
    plt.show()

    # Save / overwrite segment file
    if savepath:
        filepath = savepath / (savepath.stem + "_segments.geojson")

        if filepath.exists():
            prompt_success = False
            while not prompt_success:
                overwrite = str(input(f"Overwriting {filepath}. Proceed? (Y/N)"))
                if overwrite == "Y" or overwrite == "y":
                    prompt_success = True
                    filepath.write_text(segments.to_json())
                elif overwrite == "N" or overwrite == "n":
                    prompt_success = True
                    sys.exit("Exiting")

    return segments


if __name__ == "__main__":
    filepath = Path("data/east_york_ontario/")

    add_osm_coverage_features(
        filepath / (filepath.stem + "_segments.geojson"),
        filepath / (filepath.stem + "_osm_buildings_parks.geojson"),
    )
