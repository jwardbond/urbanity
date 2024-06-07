from pathlib import Path, PurePath
import argparse

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from genregion import generate_regions

import utils


def segment(
    network: PurePath | gpd.GeoDataFrame,
    show: bool = False,
    save: int = True,
    grid_size: int = 1024,
    area_thres: int = 10000,
    width_thres: int = 20,
    clust_width: int = 25,
    point_precision: int = 2,
):
    """Generates polygons from a simplified road network

    This is basically a wrapper for the code outlined `here <https://github.com/PaddlePaddle/PaddleSpatial/blob/main/paddlespatial/tools/genregion/README.md`. The input to this function should be a path to a geojson file containing the road network as shapely LineStrings, in the WGS84 / EPSG:4326 format.

    For more information, see:

        `A Scalable Open-Source System for Segmenting Urban Areas with Road Networks <https://dl.acm.org/doi/10.1145/3616855.3635703>` and

    Args:
        network (PurePath or GeoDataFrame): The road network to use
        show (bool, optional): If True, displays figure
        save (bool, optional): If True, saves figure / polygons
        grid_size (int, optional): Passed to genregion. Use to build a grid dictionary for searching. Defaults to 1024.
        area_thres (int, optional): Passed to genregion. The minimum area of a generated region. Defaults to 10000.
        width_thres (int, optional): Passed to genregion. The minimum ratio of area/perimeter. Defaults to 20.
        clust_width (int, optional): Passed to genregion. The threshold that helps construct the cluster.
        point_precision (int, optional): Passed to genregion. The precision of the point object while processing.
    """
    # Load and convert to EPSG:3587
    if isinstance(network, PurePath):
        gdf = gpd.read_file(network)
    elif not isinstance(network, gpd.GeoDataFrame):
        TypeError(f"Expected PurePath or GeoDataFrame, got {type(gdf)} instead.")

    gdf = gdf.to_crs("EPSG:3587")
    edges = gdf["geometry"].to_list()

    # Extract polygons
    print("Segmenting road network...")
    urban_regions = generate_regions(
        edges,
        grid_size=grid_size,
        area_thres=area_thres,
        width_thres=width_thres,
        clust_width=clust_width,
        point_precision=point_precision,
    )
    print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

    # Convert back to the correct crs
    polygons = gpd.GeoSeries(urban_regions, crs="EPSG:3587")
    polygons = polygons.to_crs("EPSG:4326")

    # Save and plot
    savepath = network_path.parents[0]

    if show:
        colors = [tuple(np.random.uniform(0, 1, 3)) for _ in urban_regions]
        polygons.plot(color=colors)

        if save:
            plt.savefig(savepath / (savepath.stem + "_segments.png"))

        plt.show()

    if save:
        filepath = savepath / (savepath.stem + "_segments.geojson")
        filepath.write_text(polygons.to_json())

    return polygons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("networkpath")
    args = parser.parse_args()

    network_path = Path(args.networkpath)
    network_path.parents[0].mkdir(exist_ok=True)

    segment(network_path, show=True)
