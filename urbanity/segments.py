import copy
from typing import Self
from pathlib import PurePath

import swifter
import shapely
import numpy as np
import geopandas as gpd
from genregion import generate_regions

import utils

# TODO add saving
# TODO add adjacency attribute


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

    def agg_features(
        self,
        polygons: gpd.GeoDataFrame | PurePath,
        feature: str,
        how: str = "mean",
        fillnan=None,
    ) -> Self:
        """Given segments and a geodataframe of polygons, aggregate the polygons on a per-segment basis.

        For example, if you have building footprints + height data, you can calculate the average height of all buildings within each segment with `how="mean"`

        Args:
            polygons (gpd.GeoDataFrame | PurePath): A geodataframe containing features and the polygons to aggregate over
            feature (str): The feature (column) name within "polygons" to aggregate
            how (str, optional): The desired aggregation behaviour. Options are "mean". Defaults to "mean".
            fillnan (_type_, optional):  Value to fill NaN entries with. Defaults to `None`.

        Raises:
            ValueError: An unsupported aggregation behaviour (`how`) was specified

        Returns:
            Self: _description_
        """

        # Parse inputs
        obj = copy.deepcopy(self)

        polygons = polygons.copy()
        polygons = utils.input_to_geodf(polygons)

        # Join
        right_gdf = polygons[["geometry", feature]]
        joined = obj.segments.sjoin(right_gdf, how="left").drop("index_right", axis=1)

        if how == "mean":
            joined = joined.groupby("id")[feature].mean()
        else:
            raise ValueError("How must be one of: mean")

        obj.segments = obj.segments.merge(joined, on="id")

        if fillnan is not None:
            obj.segments = obj.segments.fillna(value=fillnan)

        return obj

    def tiling(self, size, margin) -> Self:
        """_summary_

        Returns:
            Self: _description_
        """
        # TODO fill docstring

        # Project
        obj = copy.deepcopy(self)
        obj.segments = obj.segments.to_crs(self.proj_crs)

        # Get minimum rotated bounding rectangles
        obj.segments["mrr"] = obj.segments.minimum_rotated_rectangle()
        obj.segments["mrr_angle"] = obj.segments["mrr"].swifter.apply(
            lambda mrr: self._mrr_azimuth(mrr)
        )

        # Get a coordinate to rotate about
        bounds = obj.segments["geometry"].bounds
        obj.segments["rotation_point"] = list(zip(bounds["minx"], bounds["miny"]))

        # Rotate rectangles to be axis-aligned
        obj.segments["mrr"] = obj.segments.swifter.apply(
            lambda row: shapely.affinity.rotate(
                row.mrr, -1 * row.mrr_angle, row.rotation_point, use_radians=True
            ),
            axis=1,
        )

        # Add tiles
        obj.segments[["p1", "p2", "p3", "p4"]] = obj.segments.swifter.apply(
            lambda row: self._tile_rect(row.mrr, size, margin),
            axis=1,
            result_type="expand",
        )

        # Clip tiles to segments and select best tiling
        obj.segments["tmp_geom"] = obj.segments.swifter.apply(
            lambda row: shapely.affinity.rotate(
                row.geometry, -1 * row.mrr_angle, row.rotation_point, use_radians=True
            ),
            axis=1,
        )

        obj.segments[["best_tiling", "n_tiles"]] = obj.segments.swifter.apply(
            lambda row: self._filter_multipolygon(
                row.tmp_geom,
                [
                    row.p1,
                    row.p2,
                    row.p3,
                    row.p4,
                ],  # TODO add the other inner polys to this list
            ),
            axis=1,
            result_type="expand",
        )
        obj.segments = obj.segments.drop(labels=["p1", "p2", "p3", "p4"], axis=1)

        # Rotate the best tiling to match the original geometry
        obj.segments["best_tiling"] = obj.segments.swifter.apply(
            lambda row: shapely.affinity.rotate(
                row.best_tiling, 1 * row.mrr_angle, row.rotation_point, use_radians=True
            ),
            axis=1,
        )

        return obj

    def _mrr_azimuth(self, mrr: shapely.Polygon):
        """Calculates the azimuth of a rectangle on a cartesian plane.

        In other words. this is the angle (w.r.t. the eastern direction) that you would need to rotate and axis-aligned
        rectangle in order to get your rectangle.

        Adapted from `this SO answer <https://stackoverflow.com/questions/66108528/angle-in-minimum-rotated-rectangle>`

        Args:
            mrr (shapely.Polygon): A Shapely polygon representing a minimum rotated rectangle
        Returns:
            The azimuthal angle in radians.
        """
        bbox = list(mrr.exterior.coords)
        axis1 = np.linalg.norm(np.subtract(bbox[0], bbox[3]))
        axis2 = np.linalg.norm(np.subtract(bbox[0], bbox[1]))

        if axis1 <= axis2:
            az = self._vector_azimuth(bbox[0], bbox[1])
        else:
            az = self._vector_azimuth(bbox[0], bbox[3])

        return az

    def _vector_azimuth(self, point1: tuple, point2: tuple):
        """Calculates the azimuth between two points.

        In other words, if you are standig on the plane looking east, this is the angle you have to turn
        in order to point the same direction as the vector connecting the two points.

        Adapted from `this SO answer <https://stackoverflow.com/questions/66108528/angle-in-minimum-rotated-rectangle>`

        Returns:
            The azimuthal angle in radians.
        """

        angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
        return angle if angle > 0 else angle + np.pi

    def _tile_rect(self, mrr: shapely.Polygon, size: float, margin: float = 0):
        """Tiles an axis-aligned rectangle with squares of a given size

        Returns four different multipolygons, aligned with each corner of the minimum rotated rectangle.

        Args:
            mrr (shapely.Polygon): A Shapely polygon representing a minimum rotated rectangle
            size (Float): The size of the square to use when tiling the MRR

        Returns:
            p1, p2, p3, p4 (geopandas.GeoSeries): tiled rectangles as multipolygons.
        """
        buffered_size = size + 2 * margin

        # Tile the shape, starting with the bottom left corner
        xmin, ymin, xmax, ymax = mrr.bounds

        w = xmax - xmin
        h = ymax - ymin

        ncols, col_remainder = divmod(w, buffered_size)
        nrows, row_remainder = divmod(h, buffered_size)

        # Tile the shape, starting with the bottom left corner
        boxes = []
        for row in range(int(nrows)):
            for col in range(int(ncols)):
                bl_x = col / ncols * w + xmin + margin  # bottom left x coord of the box
                bl_y = row / nrows * h + ymin + margin
                tr_x = bl_x + size
                tr_y = bl_y + size

                box = shapely.box(
                    xmin=bl_x,
                    ymin=bl_y,
                    xmax=tr_x,
                    ymax=tr_y,
                )
                boxes.append(box)

        p1 = shapely.MultiPolygon(boxes)
        p2 = shapely.affinity.translate(p1, col_remainder)  # shift right
        p3 = shapely.affinity.translate(p1, row_remainder)  # shif up
        p4 = shapely.affinity.translate(p1, col_remainder, row_remainder)

        return p1, p2, p3, p4

    # TODO fix this.... very slow
    def _filter_multipolygon(self, bounding_poly, inner_polys):
        best_intersections = []
        for i, inner_poly in enumerate(inner_polys):
            intersections = []
            for j, p in enumerate(inner_poly.geoms):
                if bounding_poly.contains(p):
                    intersections.append(p)
            if len(intersections) > len(best_intersections):
                best_intersections = intersections

        return shapely.MultiPolygon(best_intersections), len(best_intersections)
