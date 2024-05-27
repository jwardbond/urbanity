import argparse
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

    print("Downloading boundaries...", end=" ")
    gdf_place = ox.geocoder.geocode_to_gdf(query)
    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

    polygon = gdf_place["geometry"].unary_union

    if savepath:
        filepath = savepath / (savepath.stem + "_boundary.geojson")
        filepath.write_text(shapely.to_geojson(polygon))

    return polygon


def download_osm_network(
    polygon: PurePath | shapely.Polygon, savepath: PurePath = None
):
    """Download the street network within some polygon boundary

    The coordinate system is WGS84 / EPSG:4326

    Args:
        location (str): The region to geocode to get the road network. e.g. "Toronto, Ontario, Canada"
        savepath (pathlib.PurePath, optional): Save location for downloaded polygon. Defaults to None (not saving)

    Returns:
        geojson: The road network as a geojson list of edges in WGS84 / EPSG:4326
    """

    # Parse polygon if required
    if isinstance(polygon, PurePath):
        polygon = gpd.read_file(polygon).iloc[0]["geometry"]
    elif not isinstance(polygon, shapely.Polygon):
        raise TypeError(f"Expected shapely polygon or Path, got {type(polygon)}")

    # Get graph from OSM
    print("Downloading road network...", end=" ")
    G = ox.graph_from_polygon(polygon, network_type="drive", simplify=False)
    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

    M = ox.convert.to_undirected(G)
    _, gdf = ox.convert.graph_to_gdfs(M)

    # Save and return
    output = gdf.to_json()

    if savepath:
        filepath = savepath / (savepath.stem + "_road_network.geojson")
        filepath.write_text(output)

    return output


def download_rsi(boundary_path: PurePath):  # TODO
    pass


def download_buildings(boundary_path: PurePath):  # TODO
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("savepath")
    args = parser.parse_args()

    savepath = Path(args.savepath)
    savepath.mkdir(exist_ok=True)

    download_osm_network("Kingston, Ontario", savepath)
