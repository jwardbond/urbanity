import argparse
from pathlib import Path, PurePath

import shapely
import osmnx as ox

import utils


# TODO make sure coordinates line up
# TODO Handle missing boundaries (download square box)
# TODO Query resolving, making sure they are all handled the same


def download_osm_boundary(query: str, savepath: PurePath):
    """Downloads the bounding polygon for the queried region using open streetmaps

    The coordinate system is WGS84 / EPSG:4326

    Args:
        query (str): The region to geocode to get the bounding polygon. Can be place name or address]
        savepath (pathlib.PurePath): Save location for downloaded polygon
    """

    print("Downloading boundaries...", end=" ")
    gdf_place = ox.geocoder.geocode_to_gdf(query)
    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)
    polygon = gdf_place["geometry"].unary_union

    filepath = savepath / (savepath.stem + "_boundary.geojson")
    filepath.write_text(shapely.to_geojson(polygon))


def download_osm_network(query: str, savepath: PurePath):
    """Download a street network from open street map

    The coordinate system is WGS84 / EPSG:4326

    Args:
        location (str): The region to geocode to get the road network. e.g. "Toronto, Ontario, Canada"
        savepath (pathlib.PurePath): Save location for downloaded polygon
    """

    # Get graph from OSM

    print("Downloading road network...", end=" ")
    G = ox.graph_from_place(query, network_type="drive", simplify=False)
    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)
    M = ox.convert.to_undirected(G)

    _, gdf = ox.convert.graph_to_gdfs(M)

    filepath = savepath / (savepath.stem + "_road_network.geojson")
    filepath.write_text(gdf.to_json())


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
