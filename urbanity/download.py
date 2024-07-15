import os
import argparse
import warnings
import tempfile
from pathlib import Path, PurePath

import yaml
import shapely
import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd

import utils

# The following imports are all for microsoft building footprints
# import planetary_computer
# import pystac_client
# import deltalake
import mercantile

# TODO Handle missing boundaries (download square box)


def download_osm_boundary(query: str, savefolder: PurePath = None):
    """Downloads the bounding polygon for the queried region using open streetmaps

    The coordinate system is WGS84 / EPSG:4326. For more information see `the osmnx docs <https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.geocoder.geocode_to_gdf>`

    Args:
        query (str): The region to geocode to get the bounding polygon. Can be place name or address]
        savefolder (pathlib.PurePath, optional): Save folder for downloaded polygon. Defaults to None (not saving)

    Returns:
        shapely.geometry.polygon.Polygon: A shapely polygon representing the boundary of the queried region in EPSG:4326
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
    boundary: PurePath | shapely.Polygon, savefolder: PurePath = None
):
    """Download the street network within some polygon boundary

    The coordinate system is WGS84 / EPSG:4326. For more information see `the osmnx docs <https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.graph.graph_from_polygon>`

    Args:
        boundary (PurePath, shapely.Polygon): The path to the saved boundary, or the boundary as a shapely polygon.
        savefolder (pathlib.PurePath, optional): Save location for downloaded polygons. Defaults to None (not saving)

    Returns:
        geopandas.GeoDataframe: The road network as a geojson list of edges in WGS84 / EPSG:4326
    """

    # Parse polygon if required
    boundary = _parse_polygon(boundary)

    # Get graph from OSM
    print("Downloading road network from OSM...", end=" ")
    G = ox.graph_from_polygon(boundary, network_type="drive", simplify=False)
    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

    M = ox.convert.to_undirected(G)
    _, network = ox.convert.graph_to_gdfs(M)

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
    boundary: PurePath | shapely.Polygon, savefolder: PurePath = None
):
    """Download the building polygons from open street maps within some polygon boundary

    The coordinate system is WGS84 / EPSG:4326. For more information see `the osmnx docs <https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.features.features_from_polygon>`.

    Args:
        boundary (PurePath, shapely.Polygon): The path to the saved boundary, or the boundary as a shapely polygon.
        savefolder (pathlib.PurePath, optional): Save location for downloaded polygons. Defaults to None (not saving)

    Returns:
        geopandas.GeoDataframe: A geodataframe of all buildings within the provided boundary in WGS84 / EPSG:4326
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
    buildings = buildings[
        [
            "osmid",
            "addr:housenumber",
            "addr:street",
            "addr:unit",
            "addr:postcode",
            "geometry",
        ]
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
    savefolder: PurePath = None,
    savename: str = "custom",
):
    """Download generic features from OSMwithin some polygon boundary

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


def download_rsi(boundary_path: PurePath):  # TODO complete function
    pass


def download_ms_buildings(
    boundary: PurePath | shapely.Polygon,
    savefolder: PurePath = None,
):
    """Download the building polygon data within a given boundary

    The coordinate system is WGS84 / EPSG:4326.
    Most of the code is adapted from `here <https://github.com/microsoft/GlobalMLBuildingFootprints/blob/main/examples/example_building_footprints.ipynb>`.

    Args:
        boundary (PurePath, shapely.Polygon): The path to the saved boundary, or the boundary as a shapely polygon.
        savefolder (pathlib.PurePath, optional): Save location for downloaded polygons. Defaults to None (not saving)

    Returns:
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
        "https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv"
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
                raise ValueError(f"Multiple rows found for QuadKey: {quad_key}")
            else:
                raise ValueError(f"QuadKey not found in dataset: {quad_key}")
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

    buildings = buildings.drop(labels=["id", "properties"], axis=1)
    if savefolder:
        savepath = savefolder / (savefolder.stem + "_ms_buildings.geojson")
        utils.save_geodf_with_prompt(buildings, savepath)

    buildings.insert(loc=0, column="id", value=buildings.index)

    return buildings


def _parse_polygon(
    polygon: PurePath | shapely.Polygon,
):  # TODO not sure I like this function.
    """Parses polygon representations to return a shapely polygon"""

    if isinstance(polygon, PurePath):
        polygon = gpd.read_file(polygon).iloc[0]["geometry"]
    elif not isinstance(polygon, shapely.Polygon):
        raise TypeError(f"Expected shapely polygon or Path, got {type(polygon)}")

    return polygon


# if __name__ == "__mains__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("place")
#     args = parser.parse_args()
#     place = args.place

#     savepath = Path(f"results/{_place_to_filename(place)}")

#     savepath.mkdir(exist_ok=True, parents=True)

#     print(savepath / (savepath.stem + "_filename"))
#     download_osm_boundary(query=place, savepath=savepath)

#     download_osm_network(
#         polygon=savepath / (savepath.stem + "_boundary.geojson"),
#         savepath=savepath,
#     )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("boundarypath")
    parser.add_argument
    args = parser.parse_args()
    boundarypath = Path(args.boundarypath)

    with open(Path("./data/temp_tags.yml")) as f:
        tags = yaml.safe_load(f)

    download_ms_buildings(
        boundary=boundarypath,
    )
