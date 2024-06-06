import re
import argparse
import warnings
from pathlib import Path, PurePath

import shapely
import osmnx as ox
import geopandas as gpd

import utils


# TODO make sure coordinates line up
# TODO Handle missing boundaries (download square box)
# TODO Query resolving, making sure they are all handled the same


def download_osm_boundary(query: str, savepath: PurePath = None):
    """Downloads the bounding polygon for the queried region using open streetmaps

    The coordinate system is WGS84 / EPSG:4326.

    Args:
        query (str): The region to geocode to get the bounding polygon. Can be place name or address]
        savepath (pathlib.PurePath, optional): Save location for downloaded polygon. Defaults to None (not saving)

    Returns:
        shapely.geometry.polygon.Polygon: A shapely polygon representing the boundary of the queried region in EPSG:4326
    """
    print(savepath)

    print(f"Downloading boundaries for {query} from OSM...", end=" ")
    gdf_place = ox.geocoder.geocode_to_gdf(query)
    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

    polygon = gdf_place["geometry"].unary_union

    if savepath:
        filepath = savepath / (savepath.stem + "_boundary.geojson")
        filepath.write_text(shapely.to_geojson(polygon))

    print(savepath, 3)

    return polygon


def download_osm_network(
    polygon: PurePath | shapely.Polygon, savepath: PurePath = None
):
    """Download the street network within some polygon boundary

    The coordinate system is WGS84 / EPSG:4326

    Args:
        polygon (PurePath, shapely.Polygon): The path to the saved polygon, or the polygon itself.
        savepath (pathlib.PurePath, optional): Save location for downloaded polygon. Defaults to None (not saving)

    Returns:
        geopandas: The road network as a geojson list of edges in WGS84 / EPSG:4326
    """

    # Parse polygon if required
    polygon = _parse_polygon(polygon)

    # Get graph from OSM
    print("Downloading road network from OSM...", end=" ")
    G = ox.graph_from_polygon(polygon, network_type="drive", simplify=False)
    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

    M = ox.convert.to_undirected(G)
    _, gdf = ox.convert.graph_to_gdfs(M)

    # Save and return
    if savepath:
        filepath = savepath / (savepath.stem + "_road_network.geojson")
        filepath.write_text(gdf.to_json())

    return gdf


def download_osm_buildings(
    polygon: PurePath | shapely.Polygon, savepath: PurePath = None
):
    # Parse polygon if required
    polygon = _parse_polygon(polygon)

    # Get buildings from
    print("Downloading buildings from OSM...", end=" ")

    with warnings.catch_warnings():  # HACK geopandas warning suppression
        warnings.simplefilter("ignore")
        feats = ox.features_from_polygon(polygon, tags={"building": True})

    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

    # Save and return
    if savepath:
        filepath = savepath / (savepath.stem + "osm_building_polygons.geojson")
        filepath.write_text(feats.to_json())

    print(feats.head(3))

    return feats


def download_rsi(boundary_path: PurePath):  # TODO
    pass


def download_buildings(boundary_path: PurePath):  # TODO
    pass


def _place_to_filename(place: str):
    """Given a string, strip all punctuation, lowercase, and replaces spaces with underscores"""
    place = place.lower()
    place = re.sub(r"[^\w\s]", "", place)
    place = place.replace(" ", "_")
    return place


def _parse_polygon(polygon: PurePath | shapely.Polygon):
    """Parses polygon representations to return a shapely polygon"""

    if isinstance(polygon, PurePath):
        polygon = gpd.read_file(polygon).iloc[0]["geometry"]
    elif not isinstance(polygon, shapely.Polygon):
        raise TypeError(f"Expected shapely polygon or Path, got {type(polygon)}")

    return polygon


if __name__ == "__mains__":
    parser = argparse.ArgumentParser()
    parser.add_argument("place")
    args = parser.parse_args()
    place = args.place

    savepath = Path(f"results/{_place_to_filename(place)}")

    savepath.mkdir(exist_ok=True, parents=True)

    print(savepath / (savepath.stem + "_filename"))
    download_osm_boundary(query=place, savepath=savepath)

    download_osm_network(
        polygon=savepath / (savepath.stem + "_boundary.geojson"),
        savepath=savepath,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("boundarypath")
    args = parser.parse_args()
    boundarypath = Path(args.boundarypath)

    download_osm_buildings(boundarypath, boundarypath.parents[0])
