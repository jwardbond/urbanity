import copy
import os
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import shapely
from geopandas.testing import assert_geoseries_equal

import utils
from urbanity import Region


class TestSjoinGreatestIntersection(unittest.TestCase):
    def setUp(self) -> None:
        """
        two markets m1 and m2 - each composed of two boxes
        - submarkets sub1 - sub4
        m1 consists of two squares like this:
        (-1,1) ............... (1,1)
            .      .      .
            . sub1 . sub2 .
            .      .      .
        (-1,0) ............... (1,0)
        m2 consists of two squares like this:
                    ........ (2,3)
                    .      .
                    . sub4 .
                    .      .
        (0,2) ................ (2,2)
            .              .
            .              .
            .     sub3     .
            .              .
            .              .
        (0,0) ................ (2,0)
        """
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

        # start as a projected CRS (use Mercator) for accurate area calc
        points = np.array([(-1, 0), (-1, 1), (0, 1), (0, 0), (-1, 0)])
        p1 = shapely.Polygon(points)
        p2 = shapely.Polygon(points + np.array([[1, 0]]))
        p3 = shapely.Polygon((points * 2) + np.array([2, 2]))
        p4 = shapely.Polygon(points + np.array([[3, 3]]))

        test_data = gpd.GeoDataFrame(
            data={
                "id": [0, 1, 2, 3],
                "market": ["m1", "m1", "m2", "m2"],
                "submarket": ["sub1", "sub2", "sub3", "sub4"],
            },
            geometry=[p1, p2, p3, p4],
        )
        # Define shapes in equal area spatial reference system where 1 unit is 1 metre
        test_data = test_data.set_crs("EPSG:3347")

        # Now transform to more commonly used system
        self.test_data = test_data.to_crs("EPSG:4326")
        self.test_data = test_data

    def test_sjoin_greatest_intersection_left(self):
        # change back to the original crs so numbers are easy and round
        simple_shapes = self.test_data.to_crs("EPSG:3347")

        sub2_clone = simple_shapes[simple_shapes.submarket == "sub2"].geometry.iloc[0]
        sub4_clone = simple_shapes[simple_shapes.submarket == "sub4"].geometry.iloc[0]

        # overlap sub4 with sub3
        sub3_4_overlap = shapely.affinity.translate(sub4_clone, yoff=-0.3)

        # Shift sub2 a little away from sub1
        sub2_shifted = shapely.affinity.translate(sub2_clone, xoff=0.1)

        # overlap sub2 with sub1
        gdf = gpd.GeoDataFrame(
            data={"shape_name": ["sub2", "sub3_4_overlap"]},
            geometry=[sub2_shifted, sub3_4_overlap],
        ).set_crs("EPSG:3347")

        # Our orginal data had 4 shapes. New data has 3 shapes, two of which overlap with shapes
        # from the original data

        # Left join the two geodataframes using the custom sjoin
        rslt = utils.sjoin_greatest_intersection(simple_shapes, gdf, ["shape_name"])

        # The original markets should still be there
        self.assertListEqual(
            rslt["submarket"].to_list(),
            ["sub1", "sub2", "sub3", "sub4"],
        )

        # The output should be a geodataframe
        self.assertIsInstance(rslt, gpd.GeoDataFrame)

        # Id should be a column, not an index
        self.assertTrue("id" in rslt.columns)
        self.assertFalse(rslt.index.name == "id")

        # Geometries should be preserved
        assert_geoseries_equal(rslt.geometry, simple_shapes.geometry)

        # Column names should be preserved
        self.assertTrue(all(c in rslt.columns for c in simple_shapes.columns))

        # New column should be added
        self.assertTrue("shape_name" in rslt.columns)
