import unittest
import warnings

import geopandas as gpd
import numpy as np
import shapely

from urbanity.polyronoi.longsgis import (
    densify_polygon,
    minimum_distance,
    valid_comparisons,
    voronoiDiagram4plg,
)


class TestVoronoiDiagram4Plg(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        # Create sample polygons for testing
        p1 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = shapely.Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        self.gdf = gpd.GeoDataFrame(
            data={"id": [0, 1]},
            geometry=[p1, p2],
            crs="EPSG:3347",
        )
        self.mask = shapely.Polygon([(-1, -1), (4, -1), (4, 2), (-1, 2)])

    def test_voronoi_with_densification(self):
        result = voronoiDiagram4plg(self.gdf, self.mask, densify=True)

        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result.geometry.is_valid))

    def test_voronoi_vertex_limit_exceeded(self):
        # Create polygon with many vertices
        coords = [(x / 1000, np.sin(x / 1000)) for x in range(501000)]
        poly = shapely.Polygon(coords)
        gdf_large = gpd.GeoDataFrame(geometry=[poly])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            result = voronoiDiagram4plg(gdf_large, self.mask)
        self.assertTrue(all(result.geometry.isna()))

    def test_voronoi_single_polygon(self):
        single_gdf = gpd.GeoDataFrame(
            geometry=[self.gdf.iloc[0].geometry],
            crs="EPSG:3347",
        )
        result = voronoiDiagram4plg(single_gdf, self.mask, densify=False)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 1)
        self.assertTrue(result.geometry.iloc[0].is_valid)


class TestMinimumDistance(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=DeprecationWarning)

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
    def setUp(self):
        warnings.simplefilter("ignore", category=DeprecationWarning)

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


class TestDensifyPolygon(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=DeprecationWarning)

    # def test_densify_points(self):
    #     polygons = [
    #         shapely.Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]),
    #         shapely.Polygon([(1, 1), (1, 1), (1, 1), (1, 1)]),
    #     ]
    #     gdf = gpd.GeoDataFrame(geometry=polygons)
    #     result = densify_polygon(gdf)
    #     self.assertEqual(len(result), 1)
    #     self.assertEqual(len(result.iloc[0].geometry.exterior.coords), 5)

    def test_densify_multiple_polygons_fixed_spacing(self):
        polygons = [
            shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            shapely.Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]),
        ]
        gdf = gpd.GeoDataFrame(geometry=polygons)
        result = densify_polygon(gdf, spacing=0.1)
        self.assertEqual(len(result), 2)
        self.assertGreater(len(result.iloc[0].geometry.exterior.coords), 5)
        self.assertGreater(len(result.iloc[1].geometry.exterior.coords), 5)

    def test_densify_empty_polygon(self):
        # Create an empty polygon
        empty_polygon = shapely.Polygon([])
        gdf = gpd.GeoDataFrame(geometry=[empty_polygon])

        # Test with auto spacing
        with self.assertRaises(ValueError):
            densify_polygon(gdf)


if __name__ == "__main__":
    unittest.main()
