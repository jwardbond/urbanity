import os
import sys
import unittest
import warnings
from pathlib import Path

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

import urbanity.download as ud
import utils

os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]),
    "Library",
    "share",
    "gdal",
)  # HACK GDAL warning suppression


class TestDownloadOSMBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.query = "Little Portugal, Toronto"

        cls.boundarypath = Path("./tests/test_files/test_files_boundary.geojson")
        cls.boundarypath.unlink(missing_ok=True)  # delete existing files

    def setUp(self) -> None:
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

    def test_download_osm_boundary(self) -> None:
        query = self.query
        boundarypath = self.boundarypath

        with utils.HiddenPrints():
            boundary = ud.download_osm_boundary(query, savefolder=boundarypath.parent)

        # Test that the output file exists
        self.assertTrue(boundarypath.exists())

        # Test that the output and the return value are the same
        gdf = gpd.read_file(boundarypath)
        self.assertEqual(gdf.iloc[0]["geometry"], boundary)

        # TODO test crs

    def test_download_osm_boundary_nonexistent(self):  # TODO
        pass


class DownloadOSMNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.boundarypath = Path("./tests/test_files/test_files_boundary.geojson")

        cls.networkpath = Path("./tests/test_files/test_files_road_network.geojson")
        cls.networkpath.unlink(missing_ok=True)

    def setUp(self) -> None:
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

    # delete existing files

    def test_download_from_file(self):
        networkpath = self.networkpath
        boundarypath = self.boundarypath

        with utils.HiddenPrints():
            network = ud.download_osm_network(
                boundarypath,
                savefolder=networkpath.parent,
            )

        # Test that the output file exists
        self.assertTrue(networkpath.exists())

        # Test that the crs is correct
        self.assertEqual("EPSG:4326", network.crs)

        # test that ouputs match
        network_from_file = utils.load_geojson(networkpath)
        assert_geodataframe_equal(network, network_from_file)

    def test_download_error(self):
        boundary = "East York, Toronto"

        self.assertRaises(TypeError, ud.download_osm_network, boundary)


class TestDownloadOSMBuildings(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.boundarypath = Path("./tests/test_files/test_files_boundary.geojson")

        cls.buildingspath = Path("./tests/test_files/test_files_osm_buildings.geojson")
        cls.buildingspath.unlink(missing_ok=True)

    def setUp(self) -> None:
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

    def test_download_from_file(self):
        buildingspath = self.buildingspath
        boundarypath = self.boundarypath

        with utils.HiddenPrints():
            buildings = ud.download_osm_buildings(
                boundarypath,
                savefolder=buildingspath.parent,
            )

        # Test that the output file exists
        self.assertTrue(buildingspath.exists())

        # Test that the crs is correct
        self.assertEqual("EPSG:4326", buildings.crs)

        # test that ouputs match
        buildings_from_file = utils.load_geojson(buildingspath)
        assert_geodataframe_equal(buildings, buildings_from_file)

    def test_download_error(self):
        boundary = "East York, Toronto"

        self.assertRaises(TypeError, ud.download_osm_buildings, boundary)


class TestDownloadMSBuildings(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.boundarypath = Path("./tests/test_files/test_files_boundary.geojson")

        cls.buildingspath = Path("./tests/test_files/test_files_ms_buildings.geojson")
        cls.buildingspath.unlink(missing_ok=True)

    def setUp(self) -> None:
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

    def test_download_from_file(self):
        buildingspath = self.buildingspath
        boundarypath = self.boundarypath

        with utils.HiddenPrints():
            buildings = ud.download_ms_buildings(
                boundarypath,
                savefolder=buildingspath.parent,
            )

        # Test that the output file exists
        self.assertTrue(buildingspath.exists())

        # Test that the crs is correct
        self.assertEqual("EPSG:4326", buildings.crs)

        # test that ouputs match
        buildings_from_file = utils.load_geojson(buildingspath)
        assert_geodataframe_equal(buildings, buildings_from_file)

    def test_download_error(self):
        boundary = "East York, Toronto"

        self.assertRaises(TypeError, ud.download_ms_buildings, boundary)


# TODO add test for ms buildings

if __name__ == "__main__":
    unittest.main(buffer=True)
