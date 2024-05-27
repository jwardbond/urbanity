import unittest
import warnings
from pathlib import Path

import geopandas as gpd

import urbanity.download as ud

import os
import sys

os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]), "Library", "share", "gdal"
)  # HACK

# TODO add some test cases for coordinate systems


class TestDownloadOSMBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.query = "East York, Ontario"
        cls.boundarypath = Path("./tests/test_files/test_files_boundary.geojson")
        cls.boundarypath.unlink(missing_ok=True)  # delete existing files

    def setUp(self) -> None:
        warnings.simplefilter("ignore", category=DeprecationWarning)  # HACK

    def test_download_osm_boundary(self) -> None:
        query = self.query
        boundarypath = self.boundarypath

        boundary = ud.download_osm_boundary(query, savepath=boundarypath.parent)

        # Test that the output file exists
        self.assertTrue(boundarypath.exists())

        # Test that the output and the return value are the same
        gdf = gpd.read_file(boundarypath)
        self.assertEqual(gdf.iloc[0]["geometry"], boundary)

    def test_download_osm_boundary_nonexistent(self):  # TODO
        pass


class DownloadOSMNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.simplefilter("ignore", category=DeprecationWarning)  # HACK
        cls.boundarypath = Path("./tests/test_files/test_files_boundary.geojson")
        cls.boundarypath.unlink(missing_ok=True)

        cls.boundary = ud.download_osm_boundary(
            "East York, Toronto", savepath=cls.boundarypath.parent
        )

    def setUp(self) -> None:
        warnings.simplefilter("ignore", category=DeprecationWarning)  # HACK

        self.networkpath = Path("./tests/test_files/test_files_road_network.geojson")
        self.networkpath.unlink(missing_ok=True)  # delete existing files

    def test_download_from_polygon(self):
        networkpath = self.networkpath
        boundary = self.boundary

        network = ud.download_osm_network(boundary, savepath=networkpath.parent)

        # Test that the output file exists
        self.assertTrue(networkpath.exists())

        # The rest of the functionality is tested in DownloadOSMNetwork.test_download_from_file

    def test_download_from_file(self):
        networkpath = self.networkpath
        boundarypath = self.boundarypath

        network = ud.download_osm_network(boundarypath, savepath=networkpath.parent)

        # Test that the output file exists
        self.assertTrue(networkpath.exists())

        # Test that the crs is correct
        self.assertEqual(network.crs, "EPSG:4326")

    def test_download_error(self):
        boundary = "East York, Toronto"

        self.assertRaises(TypeError, ud.download_osm_network, boundary)


# class TestDownloadOsmNetwork(unittest.TestCase):
#     def setUp(self) -> None:
#         warnings.simplefilter("ignore", category=DeprecationWarning)

#     def test_download_osm_network_exists(self) -> None:
#         query = "Kingston, Ontario"
#         outpath = Path("./temp")
#         outpath.mkdir(exist_ok=True)

#         ud.download_osm_network(query, outpath)

#         fileout = outpath / (outpath.stem + "_road_network.geojson")

#         self.assertTrue(fileout.exists())

#     def test_download_osm_boundary_nonexistent(self):  # TODO
#         pass


if __name__ == "__main__":
    unittest.main(buffer=True)
