import unittest
import warnings
from pathlib import Path

import urbanity.download as ud


class TestDownloadOsmBoundary(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter("ignore", category=DeprecationWarning)

    def test_download_osm_boundary_exists(self) -> None:
        query = "East York, Ontario"
        outpath = Path("./temp")
        outpath.mkdir(exist_ok=True)

        ud.download_osm_boundary(query, outpath)

        fileout = outpath / (outpath.stem + "_boundary.geojson")

        self.assertTrue(fileout.exists())

    def test_download_osm_boundary_nonexistent(self):  # TODO
        pass


class TestDownloadOsmNetwork(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter("ignore", category=DeprecationWarning)

    def test_download_osm_network_exists(self) -> None:
        query = "Kingston, Ontario"
        outpath = Path("./temp")
        outpath.mkdir(exist_ok=True)

        ud.download_osm_network(query, outpath)

        fileout = outpath / (outpath.stem + "_road_network.geojson")

        self.assertTrue(fileout.exists())

    def test_download_osm_boundary_nonexistent(self):  # TODO
        pass


if __name__ == "__main__":
    unittest.main(buffer=True)
