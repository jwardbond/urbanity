import os
import sys
import copy
import unittest
import warnings
from pathlib import Path


import shapely
import numpy as np
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

from urbanity import Region
import utils

os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]), "Library", "share", "gdal"
)  # HACK GDAL warning suppression


class TestRegion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.simplefilter(
            "ignore", category=DeprecationWarning
        )  # HACK geopandas warning suppression

        # Set output path and get rid of existing files
        networkpath = Path("./tests/test_files/test_files_road_network.geojson")
        cls.network = utils.input_to_geodf(networkpath)

        cls.proj_crs = "EPSG:3347"

    def test_init(self):
        network = self.network
        grid_size = 1024
        min_area = 10000
        width_thres = 20
        clust_width = 25
        point_precision = 2

        generated = Region.from_network(
            network=network,
            proj_crs=self.proj_crs,
            grid_size=grid_size,
            min_area=min_area,
            # max_area=40000,  # If you uncomment the two dataframes will not be equal
            width_thres=width_thres,
            clust_width=clust_width,
            point_precision=point_precision,
        )

        loaded = Region.load_segments(
            path_to_segments=Path("./tests/test_files/test_files_segments.geojson"),
            proj_crs=self.proj_crs,
        )

        # Test coordinate systems
        self.assertTrue("EPSG:4326", generated.segments.crs)
        self.assertTrue("EPSG:4326", loaded.segments.crs)

        # Test that the segmentation generation is (still) running correctly
        assert_geodataframe_equal(generated.segments, loaded.segments)

    def test__split_polygon(self):
        side_length = 1000

        # Wide polygons
        wide_poly = shapely.Polygon(
            (
                (0, 0),
                (side_length * 8, 0),
                (side_length * 8, side_length * 4),
                (0, side_length * 4),
            )
        )

        wide_split_left = shapely.Polygon(
            (
                (0, 0),
                (side_length * 4, 0),
                (side_length * 4, side_length * 4),
                (0, side_length * 4),
            )
        )
        wide_split_right = shapely.Polygon(
            (
                (side_length * 4, 0),
                (side_length * 8, 0),
                (side_length * 8, side_length * 4),
                (side_length * 4, side_length * 4),
            )
        )

        # Tall polygons
        tall_poly = shapely.affinity.rotate(wide_poly, 90, origin=(0, 0))
        tall_split_upper = shapely.affinity.rotate(wide_split_right, 90, origin=(0, 0))
        tall_split_lower = shapely.affinity.rotate(wide_split_left, 90, origin=(0, 0))

        split_wide = Region._split_polygon(wide_poly)
        split_tall = Region._split_polygon(tall_poly)

        # Should only divide the polygon once
        self.assertEqual(len(split_wide), 2)
        self.assertEqual(len(split_tall), 2)

        # Should divide a wide poly in half with a vertical line in the middle
        self.assertTrue(shapely.equals(wide_split_left, split_wide[0]))
        self.assertTrue(shapely.equals(wide_split_right, split_wide[1]))

        # Should divide a tall polygon in half with a horizontal line in the center
        self.assertTrue(shapely.equals(tall_split_lower, split_tall[0]))
        self.assertTrue(shapely.equals(tall_split_upper, split_tall[1]))

    def test_subdivide_segments(self):
        side_length = 10
        max_area = 100

        # Wide polygons
        poly = shapely.Polygon(
            (
                (0, 0),
                (side_length * 8, 0),
                (side_length * 8, side_length * 4),
                (0, side_length * 4),
            )
        )

        segments = gpd.GeoDataFrame(geometry=[poly, poly], crs=self.proj_crs)
        segments = Region.subdivide_segments(segments, max_area=max_area)

        # There should be 32 segments
        self.assertEqual(len(segments), 64)

        # No segment should have an area bigger than max_area
        self.assertTrue((segments["geometry"].area <= max_area).all())


class TestRegionFeatureMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.simplefilter(
            "ignore", category=DeprecationWarning
        )  # HACK geopandas warning suppression

        # Set output path and get rid of existing files
        cls.region = Region.load_segments(
            path_to_segments=Path("./tests/test_files/test_files_segments.geojson"),
            proj_crs="EPSG:3347",
        )

    def test_subtract_polygons(self) -> None:
        region = copy.deepcopy(self.region)
        polygons = utils.input_to_geodf(
            Path("./tests/test_files/test_files_osm_buildings.geojson")
        )

        new = region.subtract_polygons(polygons)

        # region should be unchanged
        self.assertTrue(region == self.region)

        # new should be different
        self.assertFalse(new == region)

        # new should have same crs
        self.assertEqual(new.proj_crs, region.proj_crs)

        # new should have WSG84 CRS
        self.assertEqual(new.segments.crs, "EPSG:4326")

        # new should have a smaller total area
        self.assertLess(
            new.segments.to_crs(new.proj_crs).area.sum(),
            region.segments.to_crs(region.proj_crs).area.sum(),
        )

    def test_agg_features(self) -> None:
        region = copy.deepcopy(self.region)
        polygons = utils.input_to_geodf(
            Path("./tests/test_files/test_files_ms_buildings.geojson")
        )

        new = region.agg_features(polygons, feature="height", how="mean", fillnan=0)

        # region should be unchanged
        self.assertTrue(region == self.region)

        # new should be different
        self.assertFalse(new == region)

        # new should have same crs
        self.assertEqual(new.proj_crs, region.proj_crs)

        # new should have WSG84 CRS
        self.assertEqual(new.segments.crs, "EPSG:4326")

        # new should have the same total area
        self.assertAlmostEqual(
            new.segments.to_crs(new.proj_crs).area.sum(),
            region.segments.to_crs(region.proj_crs).area.sum(),
        )
