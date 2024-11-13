import copy
from typing import Self
from pathlib import PurePath

import shapely
import shapely.ops
import numpy as np
import pandas as pd
import geopandas as gpd
from genregion import generate_regions

import utils
from .buildings import Buildings

# TODO add saving
# TODO add adjacency attribute
# TODO clarify docstring for returning Self / obj
# TODO refactor to create a "buildings" class


class Region:
    """The functional unit of urbanity: a region divided into neighbourhood "segments" using the road network.

    Segments are a vector-based, polygonal representation of a geographic area generated from road networks according to the code outlined `here <https://github.com/PaddlePaddle/PaddleSpatial/blob/main/paddlespatial/tools/genregion/README.md`.

    Attributes:
        segments(gpd.GeoDataFrame): A geodataframe containing (at least) the polygon segments of a given region.
        road_network(gpd.GeoDataFrame): A geodataframe containing the road network
        proj_crs(str): The crs used for anything that requires projection, the value can be anything accepted by `pyroj <https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input>` such as an authority string (eg "EPSG:4326") or a WKT string.
    """

    def __init__(
        self,
        segments: gpd.GeoDataFrame,
        proj_crs: str,
        road_network: gpd.GeoDataFrame = None,
        buildings: Buildings = None,
    ):
        if "area" not in segments:
            segments["area"] = segments.to_crs(proj_crs)["geometry"].area

        if "id" not in segments:
            segments.insert(loc=0, column="id", value=range(len(segments)))

        self.proj_crs = proj_crs
        self.segments = segments
        self.road_network = road_network
        self._buildings = buildings

    @property
    def buildings(self):
        if self._buildings is None:
            raise AttributeError("Buildings data has not been set")
        return self._buildings

    @buildings.setter
    def buildings(self, obj: Buildings):
        if type(obj) is not Buildings:
            raise TypeError("value must be a Buildings object")

        self._buildings = obj
        self._buildings.proj_crs = self.proj_crs

    @classmethod
    def build_from_network(
        cls,
        network: gpd.GeoDataFrame | gpd.GeoSeries,
        proj_crs: str,
        grid_size: int = 1024,
        min_area: int = 10000,
        max_area: int = 0,
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
            proj_crs (str): The crs used for anything that requires projection, the value can be anything accepted by `pyroj <https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input>` such as an authority string (eg "EPSG:4326") or a WKT string.
            grid_size (int, optional):Passed to _create_segments for segmentation. Used to build a grid dictionary for searching. Defaults to 1024.
            min_area (int, optional): Passed to _create_segments for segmentation. The minimum area of a generated region. Defaults to 10000.
            max_area (int, optional): Passed to _subdivide_segments for segmentation. The maximum area of a generated region. Defaults to 0 (no max area).
            width_thres (int, optional): Passed to _create_segments for segmentation. The minimum ratio of area/perimeter. Defaults to 20.
            clust_width (int, optional): Passed to _create_segments for segmentation. The threshold that helps construct the cluster.
            point_precision (int, optional): Passed to _create_segments for segmentation. The precision of the point object while processing.

        Returns:
            Region: An instance of Region with the segments in WGS84/EPSG:4326
        """

        # Convert to projected crs
        network = network.to_crs(proj_crs)
        edges = network["geometry"].to_list()

        # Extract polygons
        # print("Segmenting road network...", end=" ")
        with utils.HiddenPrints():
            urban_regions = generate_regions(
                edges,
                grid_size=grid_size,
                area_thres=min_area,
                width_thres=width_thres,
                clust_width=clust_width,
                point_precision=point_precision,
            )
        # print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

        segments = gpd.GeoDataFrame(geometry=urban_regions, crs=proj_crs)

        if max_area:
            segments = cls._subdivide_segments(segments, max_area)

        # Convert back to default crs
        segments = segments.to_crs("EPSG:4326")

        segments["id"] = segments.index
        segments = segments[["id", "geometry"]]

        return Region(segments, proj_crs, road_network=network)

    @classmethod
    def load_from_files(
        cls,
        segments_path: PurePath,
        proj_crs: str,
        road_network_path: PurePath = None,
        buildings_path: PurePath = None,
    ) -> Self:
        """Creates a Region object using saved .geojson files for the relevant attributes
        Args:
            path_to_segments (PurePath): The path to the .geojson containing pre-made segments
            proj_crs (str): The crs used for anything that requires projection, the value can be anything accepted by `pyroj <https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input>` such as an authority string (eg "EPSG:4326") or a WKT string.

        Returns:
            Region: An instance of Region with the segments in WGS84/EPSG:4326
        """
        segments = utils.load_geojson(segments_path)
        road_network = utils.load_geojson(road_network_path)

        buildings = Buildings.read_geojson(buildings_path, proj_crs)

        return cls(segments, proj_crs, road_network, buildings)

    def subtract_polygons(
        self,
        polygons: gpd.GeoDataFrame,
    ) -> Self:
        """Subtracts polygons from segments

        This is basically a wrapper for geopandas overlay diff.

        Args:
            polygons (geopandas.GeoDataframe): Geodataframe of polygons to subtract

        Returns:
            Region: Returns a new Region object
        """

        # Parse inputs
        segments = self.segments.copy()

        # Get set difference
        segments = segments.overlay(polygons, how="difference")

        return Region(
            segments=segments,
            proj_crs=self.proj_crs,
            road_network=self.road_network,
            buildings=self._buildings,
        )

    def agg_features(
        self,
        polygons: gpd.GeoDataFrame,
        feature_name: str,
        how: str = "mean",
        fillnan=None,
    ) -> Self:
        """Given segments and a geodataframe of polygons, aggregate the polygons on a per-segment basis.

        For example, if you have building footprints + height data, you can calculate the average height of all buildings within each segment with `how="mean"`.

        Args:
            polygons (gpd.GeoDataFrame | PurePath): A geodataframe containing features and the polygons to aggregate over
            feature_name (str): The feature (column) name within "polygons" to aggregate
            how (str, optional): The desired aggregation behaviour. Options are "mean". Defaults to "mean"
            fillnan (_type_, optional):  Value to fill NaN entries with. Defaults to `None`

        Raises:
            ValueError: An unsupported aggregation behaviour (`how`) was specified

        Returns:
            Region: A copy of the `Region` object after aggregation
        """

        # Parse inputs
        segments = self.segments.copy()

        polygons = polygons.copy()

        # Join
        right_gdf = polygons[["geometry", feature_name]]
        joined = segments.sjoin(right_gdf, how="left").drop("index_right", axis=1)

        if how == "mean":
            joined = joined.groupby("id")[feature_name].mean()
        else:
            raise ValueError("How must be one of: mean")

        segments = segments.merge(joined, on="id")

        if fillnan is not None:
            segments = segments.fillna(value=fillnan)

        return Region(
            segments=segments,
            proj_crs=self.proj_crs,
            road_network=self.road_network,
            buildings=self._buildings,
        )

    def disagg_features(
        self, gdf: gpd.GeoDataFrame, feature_name: str, how: str = "area"
    ) -> Self:
        # Parse inputs
        segments = self.segments.copy()
        gdf = gdf[["geometry", feature_name]]

        # Change to projected crs
        segments = segments.to_crs(self.proj_crs)
        gdf = gdf.to_crs(self.proj_crs)

        # Intersect dataframes
        if how == "area":
            gdf["area"] = gdf["geometry"].area

            # Split the gdf by segment boundaries
            split_gdf = gpd.overlay(
                gdf, segments.drop(labels=["area"], axis=1), how="intersection"
            )  # Dropping the area temporarily just helps with naming

            # Split feature proportional to area
            split_gdf["split_area"] = split_gdf["geometry"].area
            fname = f"split_{feature_name}"

            split_gdf[fname] = (
                split_gdf[feature_name] * split_gdf["split_area"] / split_gdf["area"]
            )

            # Join back to original df
            split_gdf = split_gdf[["id", fname]]
            grouped = split_gdf.groupby("id")[fname].sum()
            segments = segments.merge(grouped, on="id", how="inner")

            # change new column to original name
            segments = segments.rename(columns={fname: feature_name})

        else:
            raise ValueError(f"how = {how} is not a valid argument")

        # change back to original crs
        segments = segments.to_crs("EPSG:4326")

        return Region(
            segments=segments,
            proj_crs=self.proj_crs,
            road_network=self.road_network,
            buildings=self._buildings,
        )

    def tile_segments(self, size, margin) -> Self:
        """Tiles all segments with squares `size` x `size`, with a gap of `margin` between them.

        Every segment is tiled using a grid aligned with the major axis of that segment.

        Args:
            size: The side length of the square you want to tile with. Units will depend on `self.proj_crs`
            margin: The margin around tiles. Units will depend on `self.proj_crs`

        Returns:
            obj: A copy of the `Region` object after tiling
        """

        # Project
        segments = self.segments.copy()
        segments = segments.to_crs(self.proj_crs)

        # Get minimum rotated bounding rectangles
        segments["mrr"] = segments.minimum_rotated_rectangle()
        segments["mrr_angle"] = segments["mrr"].swifter.apply(
            lambda mrr: self._mrr_azimuth(mrr)
        )

        # Get a coordinate to rotate about
        bounds = segments["geometry"].bounds
        segments["rotation_point"] = list(zip(bounds["minx"], bounds["miny"]))

        # Rotate rectangles to be axis-aligned
        segments["mrr"] = segments.swifter.apply(
            lambda row: shapely.affinity.rotate(
                row.mrr, -1 * row.mrr_angle, row.rotation_point, use_radians=True
            ),
            axis=1,
        )

        # Add tiles
        segments[["p1", "p2", "p3", "p4"]] = segments.swifter.apply(
            lambda row: self._tile_rect(row.mrr, size, margin),
            axis=1,
            result_type="expand",
        )

        # Clip tiles to segments and select best tiling
        segments["tmp_geom"] = segments.swifter.apply(
            lambda row: shapely.affinity.rotate(
                row.geometry, -1 * row.mrr_angle, row.rotation_point, use_radians=True
            ),
            axis=1,
        )

        segments[["best_tiling", "n_tiles"]] = segments.swifter.apply(
            lambda row: self._filter_multipolygon(
                row.tmp_geom,
                [
                    row.p1,
                    row.p2,
                    row.p3,
                    row.p4,
                ],
            ),
            axis=1,
            result_type="expand",
        )
        segments = segments.drop(labels=["p1", "p2", "p3", "p4"], axis=1)

        # Rotate the best tiling to match the original geometry
        segments["best_tiling"] = segments.swifter.apply(
            lambda row: shapely.affinity.rotate(
                row.best_tiling, 1 * row.mrr_angle, row.rotation_point, use_radians=True
            ),
            axis=1,
        )

        return Region(
            segments=segments,
            proj_crs=self.proj_crs,
            road_network=self.road_network,
            buildings=self._buildings,
        )

    @classmethod
    def _subdivide_segments(
        cls, segments: gpd.GeoDataFrame, max_area: int
    ) -> gpd.GeoDataFrame:
        """Subdivides all segments greater than a minimum area.

        Overly large segments are divided in half either vertically or horizontally until they are below the max_area.

        Args:
            segments (geopandas.Geodataframe): A geodataframe containing the segments
            max_area (int): The maximum area of a segment. Units will depend on the value of `Region.proj_crs`

        Returns:
            segments (geopandas.Geodataframe): A geodataframe containing the subdivided segments.
        """
        # TODO Figure out how to handle buildings on edges.
        segments = copy.deepcopy(segments)

        larger = segments[segments.area > max_area].copy()
        smaller = segments[segments.area <= max_area].copy()

        while not larger.empty:
            # Split large geometries
            larger["geo_tmp"] = larger.apply(
                lambda row: cls._split_polygon(row.geometry), axis=1
            )
            larger = larger.explode(column="geo_tmp", ignore_index=True)
            larger["geometry"] = larger["geo_tmp"]

            # Combine the split dataframe again
            segments = gpd.GeoDataFrame(pd.concat([larger, smaller]))
            segments["area"] = segments.geometry.area

            # Generate new dataframe splits
            larger = segments[segments.area > max_area].copy()
            smaller = segments[segments.area <= max_area].copy()

        # re-index
        segments = segments.drop(labels=["geo_tmp"], axis=1)
        segments = segments.reset_index(drop=True)

        return segments

    @classmethod
    def _split_polygon(cls, geom: shapely.Polygon):
        """Splits a polygon in half either vertically or horizontally.

        Args:
            geom (shapely.Polygon): A shapely polygon

        Returns:
            geoms (list): A list of shapely polygons
        """

        bounds = geom.bounds

        # If geometry is longer than it is tall, split along a vertical line
        if (bounds[2] - bounds[0]) > (bounds[3] - bounds[1]):
            x_mid = (bounds[0] + bounds[2]) / 2
            splitter = shapely.LineString([(x_mid, bounds[1]), (x_mid, bounds[3])])

        # Else, split along a horizontal line
        else:
            y_mid = (bounds[1] + bounds[3]) / 2
            splitter = shapely.LineString([(bounds[0], y_mid), (bounds[2], y_mid)])

        # Convert from geometry collection to list
        geoms = shapely.ops.split(geom, splitter)
        geoms = list(geoms.geoms)
        return geoms

    def _mrr_azimuth(self, mrr: shapely.Polygon):
        """Calculates the azimuth of a rectangle on a cartesian plane.

        In other words. this is the angle (w.r.t. the eastern direction) that you would need to rotate an axis-aligned
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

    def __eq__(self, other: object) -> bool:
        bl = self.segments.equals(other.segments)
        bl = bl and (self.proj_crs == other.proj_crs)
        return bl
