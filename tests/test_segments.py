import os
import sys
import copy
import unittest
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

from urbanity import Segments
import utils

os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]), "Library", "share", "gdal"
)  # HACK GDAL warning suppression


class TestSegments(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.simplefilter(
            "ignore", category=DeprecationWarning
        )  # HACK geopandas warning suppression

        # Set output path and get rid of existing files
        networkpath = Path("./tests/test_files/test_files_road_network.geojson")
        cls.network = utils.input_to_geodf(networkpath)

    def test_init(self):
        network = self.network
        grid_size = 1024
        area_thres = 10000
        width_thres = 20
        clust_width = 25
        point_precision = 2

        generated = Segments.from_network(
            network=network,
            proj_crs="EPSG:3347",
            grid_size=grid_size,
            area_thres=area_thres,
            width_thres=width_thres,
            clust_width=clust_width,
            point_precision=point_precision,
        )

        loaded = Segments.load_segments(
            path_to_segments=Path("./tests/test_files/test_files_segments.geojson"),
            proj_crs="EPSG:3347",
        )

        # Test coordinate systems
        self.assertTrue("EPSG:4326", generated.segments.crs)
        self.assertTrue("EPSG:4326", loaded.segments.crs)

        # Test that the segmentation generation is (still) running correctly
        assert_geodataframe_equal(generated.segments, loaded.segments)


class TestSegmentsFeatureMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.simplefilter(
            "ignore", category=DeprecationWarning
        )  # HACK geopandas warning suppression

        # Set output path and get rid of existing files
        cls.region = Segments.load_segments(
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

        # new should have a smaller total area
        self.assertLess(
            new.segments.to_crs(new.proj_crs).area.sum(),
            region.segments.to_crs(region.proj_crs).area.sum(),
        )
