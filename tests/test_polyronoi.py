import unittest
import warnings

import geopandas as gpd
import shapely
from matplotlib import pyplot as plt

from urbanity.polyronoi.longsgis import minimum_distance, valid_comparisons


class TestMinimumDistance(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", category=UserWarning)

    def test_one_poly(self):
        polygons = gpd.GeoDataFrame(
            geometry=[
                shapely.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            ],
            crs="EPSG:3347",
        )

        min_dist = minimum_distance(polygons)

        self.assertIsNone(min_dist)

    def test_close_points_on_same_poly(self):
        polygons = gpd.GeoDataFrame(
            geometry=[
                shapely.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                shapely.Polygon([(3, 0), (3, 1), (4, 1), (4, 0)]),
            ],
            crs="EPSG:3347",
        )
        # Vertex distance within polygons = 1 unit
        # Distance between polygons = 2 units

        min_dist = minimum_distance(polygons)

        # Minimum distance should be 2.0 (distances withing the same polygon should be ignored)
        self.assertAlmostEqual(min_dist, 2.0)


class TestValidComparisons(unittest.TestCase):
    def test_empty_dict(self):
        empty_dict = {}
        result = valid_comparisons(empty_dict)
        self.assertEqual(result, [])

    def test_multi_dict(self):
        multi_dict = {0: [(0, 0), (1, 1)], 1: [(2, 2), (3, 3)], 2: [(4, 4)]}
        result = valid_comparisons(multi_dict)

        expected = [
            ((0, 0), (2, 2)),
            ((0, 0), (3, 3)),
            ((0, 0), (4, 4)),
            ((1, 1), (2, 2)),
            ((1, 1), (3, 3)),
            ((1, 1), (4, 4)),
            ((2, 2), (0, 0)),
            ((2, 2), (1, 1)),
            ((2, 2), (4, 4)),
            ((3, 3), (0, 0)),
            ((3, 3), (1, 1)),
            ((3, 3), (4, 4)),
            ((4, 4), (0, 0)),
            ((4, 4), (1, 1)),
            ((4, 4), (2, 2)),
            ((4, 4), (3, 3)),
        ]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
