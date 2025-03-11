import os
import json
import tempfile
from pathlib import PurePath

import geopandas as gpd
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
    gdf = gpd.GeoDataFrame(
        geometry=[polygon],
        crs="EPSG:4326",
    )

    if savefolder:
        savefolder.mkdir(exist_ok=True, parents=True)
        savepath = savefolder / (savefolder.stem + "_boundary")
        utils.save_geodf_with_prompt(gdf, savepath)

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

    network.insert(loc=0, column="id", value=network.index)

    # Save and return
    if savefolder:
        savepath = savefolder / (savefolder.stem + "_road_network")
        utils.save_geodf_with_prompt(network, savepath)

    return network


def download_osm_buildings(
    geom: shapely.Polygon | PurePath,
    savefolder: PurePath | None = None,
) -> gpd.GeoDataFrame:
    """Searches OSM database and returns OSM building footprints for the AOI.

    Args:
        geom (Polyhon, PurePath): Shapely geom for aoi

    Returns:
        buildings (geopandas dataframe): Set of polygons found for the aoi
    """
    geom = _parse_polygon(geom)

    # Download
    print("Downloading building footprints from OSM...", end=" ")
    try:
        # Query OSM for buildings and clip to AOI
        buildings = ox.features_from_polygon(geom.envelope, tags={"building": True})
        buildings = (
            buildings.clip(geom)
            .reset_index()
            .loc[lambda df: df["geometry"].geom_type != "Point"]
        )
    except Exception:  # noqa: BLE001
        print(utils.PrintColors.FAIL + "No fooprints" + utils.PrintColors.ENDC)
        return None
    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

    # Add ID column and return simplified dataframe
    buildings = buildings[["geometry"]]
    buildings.insert(0, "id", buildings.index)

    if savefolder:
        savepath = savefolder / (savefolder.stem + "_osm_buildings")
        utils.save_geodf_with_prompt(buildings, savepath)

    return buildings


def download_osm_generic(
    boundary: shapely.Polygon | PurePath,
    tags: dict,
    savefolder: PurePath | None = None,
    savename: str = "custom",
) -> gpd.GeoDataFrame:
    """Searches OSM database and returns OSM building footprints for the AOI.

    Args:
        geom (Polyhon, PurePath): Shapely geom for aoi

    Returns:
        buildings (geopandas dataframe): Set of polygons found for the aoi
    """
    boundary = _parse_polygon(boundary)

    print(f"Downloading {tags} from OSM...", end=" ")
    try:
        # Query OSM for buildings and clip to AOI
        downloaded = ox.features_from_polygon(boundary.envelope, tags=tags)
        downloaded = (
            downloaded.clip(boundary)
            .reset_index()
            .loc[lambda df: df["geometry"].geom_type != "Point"]
        )
    except Exception:  # noqa: BLE001
        print(utils.PrintColors.FAIL + "No Geometries" + utils.PrintColors.ENDC)
        return None
    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

    # Add ID column and return simplified dataframe
    downloaded = downloaded[["geometry"]]
    downloaded.insert(0, "id", downloaded.index)

    if savefolder:
        savepath = savefolder / (savefolder.stem + f"_osm_{savename}")
        utils.save_geodf_with_prompt(downloaded, savepath)

    return downloaded


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
            if rows.shape[0] >= 1:
                for _, r in rows.iterrows():
                    url = r["Url"]

                    df2 = pd.read_json(url, lines=True)
                    df2["geometry"] = df2["geometry"].apply(shapely.geometry.shape)

                    gdf = gpd.GeoDataFrame(df2, crs=4326)
                    fn = os.path.join(tmpdir, f"{quad_key}.geojson")
                    tmp_fns.append(fn)
                    if not os.path.exists(fn):
                        gdf.to_file(fn, driver="GeoJSON")
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

    # Extract height and confidence from properties
    buildings["properties"] = buildings["properties"].apply(json.loads)
    buildings["height"] = buildings["properties"].apply(lambda x: x.get("height", None))
    buildings["confidence"] = buildings["properties"].apply(
        lambda x: x.get("confidence", None)
    )

    # Output formatting
    buildings = buildings[["type", "height", "confidence", "geometry"]]
    buildings = buildings.reset_index()
    buildings.insert(loc=0, column="id", value=buildings.index)

    if savefolder:
        savepath = savefolder / (savefolder.stem + "_ms_buildings")
        utils.save_geodf_with_prompt(buildings, savepath)

    return buildings


def _parse_polygon(
    polygon: PurePath | shapely.Polygon,
):
    """Parses polygon representations to return a shapely polygon."""
    if isinstance(polygon, PurePath):
        polygon = gpd.read_parquet(polygon).iloc[0]["geometry"]
    elif not isinstance(polygon, shapely.Polygon):
        msg = f"Expected shapely polygon or Path, got {type(polygon)}"
        raise TypeError(msg)

    return polygon


# def get_google_building_footprints(geom, country_iso):
#     """Searches the Source Coop s3 buckets and returns Google footprints for the AOI.

#     Source: https://beta.source.coop/repositories/cholmes/google-open-buildings/description

#     Args:
#         geom (polygon): Shapely geom for aoi
#         country_iso (str): Country ISO string

#     Returns:
#         buildings (geopandas dataframe): Set of polygons found for the aoi
#     """
#     s3_path = f"s3://us-west-2.opendata.source.coop/google-research-open-buildings/v3/geoparquet-by-country/country_iso={country_iso}/{country_iso}.parquet"
#     s3 = s3fs.S3FileSystem(anon=True)
#     try:
#         all_buildings = pq.ParquetDataset(s3_path, filesystem=s3).read().to_pandas()
#     except Exception as err:
#         print(f"No footprints for {err}")
#         return None
#     all_buildings.geometry = all_buildings.geometry.apply(
#         lambda x: shapely.wkb.loads(x)
#     )
#     all_buildings = gpd.GeoDataFrame(all_buildings, geometry=all_buildings.geometry)
#     buildings = all_buildings.clip(geom)
#     return buildings


# ***************************************
# DEPRECATED
# ***************************************

# def download_osm_buildings(
#     boundary: PurePath | shapely.Polygon,
#     savefolder: PurePath | None = None,
# ) -> gpd.GeoDataFrame:
#     """Download the building polygons from open street maps within some polygon boundary.

#     The coordinate system is WGS84 / EPSG:4326. For more information see `the osmnx docs <https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.features.features_from_polygon>`.

#     Args:
#         boundary (PurePath, shapely.Polygon): The path to the saved boundary, or the boundary as a shapely polygon.
#         savefolder (pathlib.PurePath, optional): Save location for downloaded polygons. Defaults to None (not saving)

#     Returns:
#         buildings (geopandas.GeoDataframe): A geodataframe of all buildings within the provided boundary in WGS84 / EPSG:4326
#     """
#     # Parse polygon if required
#     boundary = _parse_polygon(boundary)

#     # Get buildings polygons from OSM
#     print("Downloading buildings from OSM...", end=" ")
#     with warnings.catch_warnings():  # HACK geopandas warning suppression
#         warnings.simplefilter("ignore")
#         buildings = ox.features_from_polygon(boundary, tags={"building": True})
#     print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

#     # Correctly format output
#     buildings = buildings.reset_index()
#     buildings.index = buildings.index.astype(np.int64)

#     # The next two lines of code are rather needlessly complicated and just
#     # select only desired columns from the gdf, while allowing for the condition
#     # that one or more of the desired columns might not exist. It then makes sure
#     # the columns are in the correct order
#     filt = [
#         "osmid",
#         "addr:housenumber",
#         "addr:street",
#         "addr:unit",
#         "addr:postcode",
#         "geometry",
#     ]
#     buildings = buildings[
#         sorted(buildings.columns.intersection(filt), key=lambda x: filt.index(x))
#     ]

#     buildings.insert(loc=0, column="id", value=buildings.index)

#     # Save and return
#     if savefolder:
#         savepath = savefolder / (savefolder.stem + "_osm_buildings")
#         utils.save_geodf_with_prompt(buildings, savepath)

#     return buildings


# def download_osm_generic(
#     boundary: PurePath | shapely.Polygon,
#     tags: dict,
#     savefolder: PurePath | None = None,
#     savename: str = "custom",
# ) -> gpd.GeoDataFrame:
#     """Download generic features from OSMwithin some polygon boundary.

#     The coordinate system is WGS84 / EPSG:4326. For more information see `the osmnx docs <https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.features.features_from_polygon>`.

#     Args:
#         boundary (PurePath, shapely.Polygon): The path to the saved boundary polygon, or the polygon itself.
#         tags (dict): A dict of tag-value combinations. See `here <https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.features.features_from_polygon>` for more details on format.
#         savefolder (pathlib.PurePath, optional): Save location for downloaded polygons. Defaults to None (not saving)
#         savename (str): name to append to saved geojson

#     Returns:
#         geopandas.GeoDataframe: A geodataframe of all features within the provided boundary in WGS84 / EPSG:4326
#     """
#     # Parse polygon if required
#     boundary = _parse_polygon(boundary)

#     # Download from osm
#     print(f"Downloading {tags} from OSM...", end=" ")

#     with warnings.catch_warnings():  # HACK geopandas warning suppression
#         warnings.simplefilter("ignore")
#         gdf = ox.features_from_polygon(boundary, tags=tags)

#     print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

#     # Remove any point geometries
#     gdf = gdf[gdf["geometry"].geom_type != "Point"]

#     # Output formatting
#     gdf = gdf.reset_index()
#     gdf.index = gdf.index.astype(np.int64)

#     # TODO prune columns
#     filt = [
#         "osmid",
#         "addr:housenumber",
#         "addr:street",
#         "addr:unit",
#         "addr:postcode",
#         "geometry",
#     ]
#     gdf = gdf[sorted(gdf.columns.intersection(filt), key=lambda x: filt.index(x))]

#     gdf.insert(loc=0, column="id", value=gdf.index)

#     # Save and return
#     if savefolder:
#         savepath = savefolder / (savefolder.stem + f"_osm_{savename}")
#         utils.save_geodf_with_prompt(gdf, savepath)

#     return gdf


if __name__ == "__main__":
    pass
