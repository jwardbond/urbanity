import os
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

import urbanity.segment as sg
import utils

os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]), "Library", "share", "gdal"
)  # HACK GDAL warning suppression


class TestSegment(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Set output path and get rid of existing files
        cls.segmentpath = Path("./tests/test_files/test_files_segments.geojson")
        cls.segmentpath.unlink(missing_ok=True)

        cls.networkpath = Path("./tests/test_files/test_files_road_network.geojson")

    def setUp(self) -> None:
        warnings.simplefilter(
            "ignore", category=DeprecationWarning
        )  # HACK geopandas warning suppression

    def test_segment(self) -> None:
        segmentpath = self.segmentpath
        networkpath = self.networkpath

        with utils.HiddenPrints():
            segments = sg.segment(networkpath, savepath=segmentpath.parent)

        # test that file was created
        self.assertTrue(segmentpath.exists())

        # test that the output is in the right crs
        self.assertEqual("EPSG:4326", segments.crs)

        # test that ouputs match
        segments_from_file = gpd.read_file(segmentpath)
        segments_from_file.set_crs("EPSG:4326")
        segments_from_file["id"] = segments_from_file["id"].astype(str).astype(np.int64)
        assert_geodataframe_equal(segments, segments_from_file)
