from pathlib import Path, PurePath

import geopandas as gpd

import utils


def subtract_polygons(
    segments: gpd.GeoDataFrame | PurePath,
    polygons: gpd.GeoDataFrame | PurePath,
    savefolder: PurePath = None,
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

    segments = utils.input_to_geodf(segments)
    polygons = utils.input_to_geodf(polygons)

    # Get set difference
    segments = segments.overlay(polygons, how="difference")

    # Save / overwrite segment file
    if savefolder:
        savepath = savefolder / (savefolder.stem + "_segments.geojson")
        utils.save_geodf_with_prompt(segments, savepath)

    return segments


def agg_features(
    segments: gpd.GeoDataFrame | PurePath,
    polygons: gpd.GeoDataFrame | PurePath,
    feature: str,
    how="mean",
    fillnan=None,
    savefolder: PurePath = None,
):
    """Given segments and a list of polygons representing areas to NOT include, add various metrics

    Args:
        segments (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of segments or path to a .geojson containing them.
        polygons (geopandas.GeoDataframe, pathlib.PurePath):
        savefolder (pathlib.PurePath, optional): Folder to save result to. Defaults to None (not saved)

    Returns:
        geopandas.GeoDataframe: A geodataframe containing the set difference segments - polygons
    """
    # TODO fill out docstring

    # Parse inputs
    segments = segments.copy()
    polygons = polygons.copy()

    segments = utils.input_to_geodf(segments)
    polygons = utils.input_to_geodf(polygons)

    # Join
    right_gdf = polygons[["geometry", feature]]
    joined = segments.sjoin(right_gdf, how="left").drop("index_right", axis=1)

    if how == "mean":
        joined = joined.groupby("id")[feature].mean()
    else:
        raise ValueError("How must be one of: mean")

    segments = segments.merge(joined, on="id")

    if fillnan is not None:
        segments = segments.fillna(value=fillnan)

    # Save / overwrite segment file
    if savefolder:
        savepath = savefolder / (savefolder.stem + "_segments.geojson")
        utils.save_geodf_with_prompt(segments, savepath)

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

    segments = utils.input_to_geodf(segments)
    buildings = utils.input_to_geodf(buildings)

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
        utils.save_geodf_with_prompt(segments, savepath)

    return segments


if __name__ == "__main__":
    filepath = Path("data/east_york_ontario/")

    add_building_features(
        filepath / (filepath.stem + "_segments.geojson"),
        filepath / (filepath.stem + "_osm_buildings_parks.geojson"),
    )
