import unittest
import warnings
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.testing import assert_geoseries_equal
from shapely.geometry import Point

from urbanity.geoparallel.geoparallel import GeoParallel


def double_x_coordinate(geom):
    return Point(geom.x * 2, geom.y)


def identity_function(x):
    return x


class TestGeoParallel(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )

        # Create sample GeoSeries for testing
        points = [Point(x, y) for x, y in zip(range(10), range(10), strict=True)]
        self.test_gs = gpd.GeoSeries(points)

    def test_geoparallel_apply(self):
        gp = GeoParallel(n_jobs=2)

        result = gp.apply(self.test_gs, double_x_coordinate)
        expected = self.test_gs.map(double_x_coordinate)

        # Output should be the same length
        self.assertEqual(len(result), len(self.test_gs))
        assert_geoseries_equal(result, expected)

    def test_geoparallel_apply_chunked(self):
        gp = GeoParallel(n_jobs=2)
        result = gp.apply_chunked(self.test_gs, double_x_coordinate, chunk_size=3)
        expected = self.test_gs.map(double_x_coordinate)

        pd.testing.assert_series_equal(result, expected, check_dtype=False)
        # assert_geoseries_equal(result, expected)

    @patch("urbanity.geoparallel.geoparallel.tqdm")
    def test_progress_bar_display(self, mock_tqdm):
        gp = GeoParallel(n_jobs=2, prog_bar=True)
        gp.apply(self.test_gs, identity_function)
        mock_tqdm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
