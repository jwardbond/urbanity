import os
import tempfile
import warnings
from pathlib import PurePath

import geopandas as gpd

# The following imports are all for microsoft building footprints
import mercantile
import numpy as np
import osmnx as ox
import pandas as pd
import shapely

import utils

# TODO Handle missing boundaries (download square box)


def download_osm_boundary(
    query: str,
    savefolder: PurePath | None = None,
) -> shapely.Polygon:
    """Downloads the bounding polygon for the queried region using open streetmaps.

    The coordinate system is WGS84 / EPSG:4326. For more information see
    `the osmnx docs <https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.geocoder.geocode_to_gdf>`

    Args:
        query (str): The region to geocode to get the bounding polygon. Can be place name or address]
        savefolder (pathlib.PurePath, optional): Save folder for downloaded polygon. Defaults to None (not saving)

    Returns:
        polygon (shapely.Polygon): A shapely polygon representing the boundary of the queried region in EPSG:4326
    """
    print(f"Downloading boundaries for {query} from OSM...", end=" ")
    gdf_place = ox.geocoder.geocode_to_gdf(query)
    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

    polygon = gdf_place["geometry"].unary_union

    if savefolder:
        savefolder.mkdir(exist_ok=True, parents=True)
        savepath = savefolder / (savefolder.stem + "_boundary.geojson")
        savepath.write_text(shapely.to_geojson(polygon))

    return polygon


def download_osm_network(
    boundary: PurePath | shapely.Polygon,
    savefolder: PurePath | None = None,
) -> gpd.GeoDataFrame:
    """Download the street network within some polygon boundary.

    The coordinate system is WGS84 / EPSG:4326. For more information see `the osmnx docs <https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.graph.graph_from_polygon>`

    Args:
        boundary (PurePath, shapely.Polygon): The path to the saved boundary, or the boundary as a shapely polygon.
        savefolder (pathlib.PurePath, optional): Save location for downloaded polygons. Defaults to None (not saving)

    Returns:
        network (geopandas.GeoDataframe): The road network as a geodataframe list of edges in WGS84 / EPSG:4326
    """
    # Parse polygon if required
    boundary = _parse_polygon(boundary)

    # Get graph from OSM
    print("Downloading road network from OSM...", end=" ")
    g = ox.graph_from_polygon(boundary, network_type="drive", simplify=False)
    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

    m = ox.convert.to_undirected(g)
    _, network = ox.convert.graph_to_gdfs(m)

    # Correctly format output
    network = network.reset_index()
    network.index = network.index.astype(np.int64)
    network = network[["geometry"]]

    # Save and return
    if savefolder:
        savepath = savefolder / (savefolder.stem + "_road_network.geojson")
        utils.save_geodf_with_prompt(network, savepath)

    network.insert(loc=0, column="id", value=network.index)

    return network


def download_osm_buildings(
    boundary: PurePath | shapely.Polygon,
    savefolder: PurePath | None = None,
) -> gpd.GeoDataFrame:
    """Download the building polygons from open street maps within some polygon boundary.

    The coordinate system is WGS84 / EPSG:4326. For more information see `the osmnx docs <https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.features.features_from_polygon>`.

    Args:
        boundary (PurePath, shapely.Polygon): The path to the saved boundary, or the boundary as a shapely polygon.
        savefolder (pathlib.PurePath, optional): Save location for downloaded polygons. Defaults to None (not saving)

    Returns:
        buildings (geopandas.GeoDataframe): A geodataframe of all buildings within the provided boundary in WGS84 / EPSG:4326
    """
    # Parse polygon if required
    boundary = _parse_polygon(boundary)

    # Get buildings polygons from OSM
    print("Downloading buildings from OSM...", end=" ")
    with warnings.catch_warnings():  # HACK geopandas warning suppression
        warnings.simplefilter("ignore")
        buildings = ox.features_from_polygon(boundary, tags={"building": True})
    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

    # Remove any point geometries
    buildings = buildings[buildings["geometry"].geom_type != "Point"]

    # Correctly format output
    buildings = buildings.reset_index()
    buildings.index = buildings.index.astype(np.int64)

    # The next two lines of code are rather needlessly complicated and just
    # select only desired columns from the gdf, while allowing for the condition
    # that one or more of the desired columns might not exist. It then makes sure
    # the columns are in the correct order
    filt = [
        "osmid",
        "addr:housenumber",
        "addr:street",
        "addr:unit",
        "addr:postcode",
        "geometry",
    ]
    buildings = buildings[
        sorted(buildings.columns.intersection(filt), key=lambda x: filt.index(x))
    ]

    # Save and return
    if savefolder:
        savepath = savefolder / (savefolder.stem + "_osm_buildings.geojson")
        utils.save_geodf_with_prompt(buildings, savepath)

    buildings.insert(loc=0, column="id", value=buildings.index)

    return buildings


def download_osm_generic(
    boundary: PurePath | shapely.Polygon,
    tags: dict,
    savefolder: PurePath | None = None,
    savename: str = "custom",
) -> gpd.GeoDataFrame:
    """Download generic features from OSMwithin some polygon boundary.

    The coordinate system is WGS84 / EPSG:4326. For more information see `the osmnx docs <https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.features.features_from_polygon>`.

    Args:
        boundary (PurePath, shapely.Polygon): The path to the saved boundary polygon, or the polygon itself.
        tags (dict): A dict of tag-value combinations. See `here <https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.features.features_from_polygon>` for more details on format.
        savefolder (pathlib.PurePath, optional): Save location for downloaded polygons. Defaults to None (not saving)
        savename (str): name to append to saved geojson

    Returns:
        geopandas.GeoDataframe: A geodataframe of all features within the provided boundary in WGS84 / EPSG:4326
    """
    # Parse polygon if required
    boundary = _parse_polygon(boundary)

    # Download from osm
    print(f"Downloading {tags} from OSM...", end=" ")

    with warnings.catch_warnings():  # HACK geopandas warning suppression
        warnings.simplefilter("ignore")
        gdf = ox.features_from_polygon(boundary, tags=tags)

    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

    # Remove any point geometries
    gdf = gdf[gdf["geometry"].geom_type != "Point"]

    # Output formatting
    gdf = gdf.reset_index()
    gdf.index = gdf.index.astype(np.int64)
    # TODO prune columns

    # Save and return
    if savefolder:
        savepath = savefolder / (savefolder.stem + f"_osm_{savename}.geojson")
        utils.save_geodf_with_prompt(gdf, savepath)

    gdf.insert(loc=0, column="id", value=gdf.index)
    return gdf


def download_ms_buildings(
    boundary: PurePath | shapely.Polygon,
    savefolder: PurePath | None = None,
) -> gpd.GeoDataFrame:
    """Download the building polygon data within a given boundary.

    The coordinate system is WGS84 / EPSG:4326.
    Most of the code is adapted from `here <https://github.com/microsoft/GlobalMLBuildingFootprints/blob/main/examples/example_building_footprints.ipynb>`.

    Args:
        boundary (PurePath, shapely.Polygon): The path to the saved boundary, or the boundary as a shapely polygon.
        savefolder (pathlib.PurePath, optional): Save location for downloaded polygons. Defaults to None (not saving)

    Returns:
        buildings (geopandas.GeoDataFrame): A geodataframe of building polygons in EPSG:4326
    """
    # Parse polygon if required
    boundary = _parse_polygon(boundary)

    # TODO verify boundary is in the correct crs

    # Set up dataset
    quad_keys = [
        int(mercantile.quadkey(tile))
        for tile in mercantile.tiles(*boundary.bounds, zooms=9)
    ]

    df = pd.read_csv(
        "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv",
    )

    # Load tiles to temp dir
    idx = 0
    buildings = gpd.GeoDataFrame()

    print("Downloading buildings from ms-buildings...", end=" ")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download the GeoJSON files for each tile that intersects the input geometry
        tmp_fns = []
        for quad_key in quad_keys:
            rows = df[df["QuadKey"] == quad_key]
            if rows.shape[0] == 1:
                url = rows.iloc[0]["Url"]

                df2 = pd.read_json(url, lines=True)
                df2["geometry"] = df2["geometry"].apply(shapely.geometry.shape)

                gdf = gpd.GeoDataFrame(df2, crs=4326)
                fn = os.path.join(tmpdir, f"{quad_key}.geojson")
                tmp_fns.append(fn)
                if not os.path.exists(fn):
                    gdf.to_file(fn, driver="GeoJSON")
            elif rows.shape[0] > 1:
                msg = f"Multiple rows found for QuadKey: {quad_key}"
                raise ValueError(msg)
            else:
                msg = f"QuadKey not found in dataset: {quad_key}"
                raise ValueError(msg)
        print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

        # Merge the GeoJSON files into a single file
        for fn in tmp_fns:
            gdf = gpd.read_file(fn)  # Read each file into a GeoDataFrame
            gdf = gdf[gdf.geometry.within(boundary)]  # Filter geometries within the AOI
            gdf["id"] = range(idx, idx + len(gdf))  # Update 'id' based on idx
            idx += len(gdf)
            buildings = pd.concat([buildings, gdf], ignore_index=True)

    # Output formatting
    # buildings = buildings.clip(boundary)
    buildings.reset_index()
    buildings.index = buildings.index.astype(np.int64)
    buildings["height"] = buildings["properties"].apply(lambda x: x["height"])
    buildings["confidence"] = buildings["properties"].apply(lambda x: x["confidence"])

    buildings = buildings[["type", "height", "confidence", "geometry"]]

    if savefolder:
        savepath = savefolder / (savefolder.stem + "_ms_buildings.geojson")
        utils.save_geodf_with_prompt(buildings, savepath)

    buildings.insert(loc=0, column="id", value=buildings.index)

    return buildings


def _parse_polygon(
    polygon: PurePath | shapely.Polygon,
):
    """Parses polygon representations to return a shapely polygon."""
    if isinstance(polygon, PurePath):
        polygon = gpd.read_file(polygon).iloc[0]["geometry"]
    elif not isinstance(polygon, shapely.Polygon):
        msg = f"Expected shapely polygon or Path, got {type(polygon)}"
        raise TypeError(msg)

    return polygon


if __name__ == "__main__":
    pass
