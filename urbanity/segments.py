import copy
from typing import Self
from pathlib import PurePath

import geopandas as gpd
from genregion import generate_regions

import utils


class Segments:
    """The functional unit of urbanity: a region divided into neighbourhood "segments" using the road network.

    Segments are a vector-based, polygonal representation of a geographic area generated from road networks according to the code outlined `here <https://github.com/PaddlePaddle/PaddleSpatial/blob/main/paddlespatial/tools/genregion/README.md`.

    Attributes:
        segments(gpd.GeoDataFrame): A geodataframe containing (at least) the polygon segments of a given region.
        proj_crs(str): The crs used for anything that requires projection, the value can be anything accepted by `pyroj <https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input>` such as an authority string (eg "EPSG:4326") or a WKT string.
    """

    def __init__(self, segments: gpd.GeoDataFrame, proj_crs: str):
        self.segments = segments
        self.proj_crs = proj_crs

    @classmethod
    def from_network(
        cls,
        network: gpd.GeoDataFrame | gpd.GeoSeries | PurePath,
        proj_crs: str,
        grid_size: int = 1024,
        area_thres: int = 10000,
        width_thres: int = 20,
        clust_width: int = 25,
        point_precision: int = 2,
    ) -> Self:
        """Creates segments from a road network.

        Creates a Segments object from a simplified road network. This is basically a wrapper for the code outlined `here <https://github.com/PaddlePaddle/PaddleSpatial/blob/main/paddlespatial/tools/genregion/README.md`. The betwork must be in the WGS84 / EPSG:4326 crs.

        For more information, see:

            `A Scalable Open-Source System for Segmenting Urban Areas with Road Networks <https://dl.acm.org/doi/10.1145/3616855.3635703>` and

        Args:
            network (GeoDataFrame or GeoSeries or Purepath): The road network to use, stored as a geodataframe/geoseries of linestrings OR the path to a .geojson containing the same.
            proj_crs(str): The crs used for anything that requires projection, the value can be anything accepted by `pyroj <https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input>` such as an authority string (eg "EPSG:4326") or a WKT string.
            grid_size (int, optional):Passed to _create_segments for segmentation. Used to build a grid dictionary for searching. Defaults to 1024.
            area_thres (int, optional): Passed to _create_segments for segmentation. The minimum area of a generated region. Defaults to 10000.
            width_thres (int, optional): Passed to _create_segments for segmentation. The minimum ratio of area/perimeter. Defaults to 20.
            clust_width (int, optional): Passed to _create_segments for segmentation. The threshold that helps construct the cluster.
            point_precision (int, optional): Passed to _create_segments for segmentation. The precision of the point object while processing.

        Returns:
            Segments: An instance of Segments with the segments in WGS84/EPSG:4326
        """

        if isinstance(network, PurePath):
            network = utils.input_to_geodf(network)

        network = network.to_crs(proj_crs)
        edges = network["geometry"].to_list()

        # Extract polygons
        print("Segmenting road network...", end=" ")
        with utils.HiddenPrints():
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
        segments = gpd.GeoDataFrame(geometry=urban_regions, crs=proj_crs)
        segments = segments.to_crs("EPSG:4326")

        segments["id"] = segments.index
        segments = segments[["id", "geometry"]]

        return cls(segments, proj_crs)

    @classmethod
    def load_segments(cls, path_to_segments: PurePath, proj_crs: str) -> Self:
        """Loads pre-existing segments from a .geojson file

        Args:
            path_to_segments (PurePath): A path
            proj_crs(str): The crs used for anything that requires projection, the value can be anything accepted by `pyroj <https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input>` such as an authority string (eg "EPSG:4326") or a WKT string.

        Returns:
            Segments: An instance of Segments
        """
        segments = utils.input_to_geodf(path_to_segments)
        return cls(segments, proj_crs)

    def __eq__(self, other: object) -> bool:
        bl = self.segments.equals(other.segments)
        bl = bl and (self.proj_crs == other.proj_crs)
        return bl

    def subtract_polygons(
        self,
        polygons: gpd.GeoDataFrame | PurePath,
    ) -> Self:
        """Subtract polygons from segments

        This is basically a wrapper for geopandas overlay diff.

        Args:
            polygons (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of polygons to subtract, or path to a .geojson containing them.

        Returns:
            Segments: Returns Segments with the polygons subtracted
        """

        # Parse inputs
        obj = copy.deepcopy(self)

        polygons = utils.input_to_geodf(polygons)
        polygons = polygons.copy(deep=True)

        # Get set difference
        obj.segments = obj.segments.overlay(polygons, how="difference")

        return obj
