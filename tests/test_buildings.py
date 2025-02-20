import copy
import os
import shutil
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal

from urbanity import Buildings
from urbanity.buildings import _dissolve_overlaps, _shrink_buildings

os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]),
    "Library",
    "share",
    "gdal",
)  # HACK GDAL warning suppression


class TestBuildingsInitialiazation(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

        # Create a mock region

        building_data = {
            "id": [0, 1, 22, 31],
            "height": [-1.0, 20.0, 12.0, 25.0],
            "geometry": [
                shapely.Polygon([(0, 0), (5, 0), (5, 10), (0, 10)]),  # A=50, V=500
                shapely.Polygon(
                    [(-5, -20), (-13, -20), (-13, -10), (-5, -10)],
                ),  # A=80, V=1600
                shapely.Polygon([(13, 0), (19, 0), (19, 10), (13, 10)]),  # A=60, V=720
                shapely.Polygon(
                    [(19, -30), (28, -30), (28, -20), (19, -20)],
                ),  # A=90, V=2250
            ],
        }

        self.proj_crs = "EPSG:3347"
        self.data = gpd.GeoDataFrame(building_data, crs=self.proj_crs)
        self.save_folder = Path(__file__).parent / "test_files" / "test_buildings_save"

    def test_init(self) -> None:
        buildings = Buildings(data=self.data, proj_crs=self.proj_crs)

        self.assertTrue("area" in buildings.data)
        self.assertTrue("id" in buildings.data)

        # Area calcs should be correct
        self.assertAlmostEqual(90, buildings.data.iloc[3]["area"], places=5)

        # Should be in "EPSG:4326"
        self.assertEqual(buildings.data.crs, "EPSG:4326")

    @patch("pathlib.Path.mkdir")
    @patch("utils.save_geodf")
    def test_save_only_one_geom(self, mock_save, mock_mkdir) -> None:  # noqa: ANN001, ARG002
        buildings = Buildings(data=self.data, proj_crs=self.proj_crs)

        buildings.save(self.save_folder)
        args, _ = mock_save.call_args
        gdf, path = args

        # Utils save should only be called once
        self.assertEqual(mock_save.call_count, 1)

        # The dataframe should have the right name
        self.assertEqual(path.name, "buildings")

        # The dataframe should have only one geometry column
        self.assertEqual(
            sum(
                isinstance(gdf[c].iloc[0], shapely.geometry.base.BaseGeometry)
                for c in gdf.columns
            ),
            1,
        )

    def test_save_and_load(self) -> None:
        buildings = Buildings(data=self.data, proj_crs=self.proj_crs)

        if self.save_folder.exists():
            shutil.rmtree(self.save_folder)

        buildings.save(self.save_folder)
        loaded = Buildings.load(self.save_folder, proj_crs="EPSG:3347")

        # Files should be created
        self.assertTrue(Path(self.save_folder / "buildings.parquet").exists())

        # Data should be unchanged
        assert_geodataframe_equal(loaded.data, buildings.data, check_like=True)


class TestBuildingsCreate(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

        # Create sample data with mix of valid/invalid geometries
        self.valid_poly = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        self.invalid_poly = shapely.Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
        self.multi_poly = shapely.MultiPolygon(
            [
                shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
                shapely.Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]),
            ],
        )
        self.empty_poly = shapely.Polygon([])

        data = {
            "geometry": [
                self.valid_poly,
                self.invalid_poly,
                self.multi_poly,
                self.empty_poly,
            ],
        }
        self.gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        self.proj_crs = "EPSG:3347"

    def test_explode_and_prune_invalid_geometries(self):
        buildings = Buildings.create(self.gdf, self.proj_crs)

        # Should only have 2 valid single polygons (1 original + 1 from multipolygon)
        self.assertEqual(len(buildings.data), 2)

        # Verify no invalid/empty geometries remain
        self.assertTrue(buildings.data.is_valid.all())
        self.assertFalse(buildings.data.is_empty.any())

        # Verify no multipolygons remain
        self.assertTrue(
            all(isinstance(geom, shapely.Polygon) for geom in buildings.data.geometry),
        )

    def test_crs_transformations(self):
        buildings = Buildings.create(self.gdf, self.proj_crs)

        # Final CRS should be EPSG:4326
        self.assertEqual(buildings.data.crs, "EPSG:4326")

    def test_buildings_instance_creation(self):
        buildings = Buildings.create(self.gdf, self.proj_crs)

        # Verify instance type and attributes
        self.assertIsInstance(buildings, Buildings)
        self.assertEqual(buildings.proj_crs, self.proj_crs)
        self.assertIsInstance(buildings.data, gpd.GeoDataFrame)


class TestBuildingsMethods(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

        # Create a mock region

        building_data = {
            "id": [0, 1, 22, 31],
            "height": [-1.0, 20.0, 12.0, 25.0],
            "geometry": [
                shapely.Polygon([(0, 0), (5, 0), (5, 10), (0, 10)]),  # A=50, V=500
                shapely.Polygon(
                    [(-5, -20), (-13, -20), (-13, -10), (-5, -10)],
                ),  # A=80, V=1600
                shapely.Polygon([(13, 0), (19, 0), (19, 10), (13, 10)]),  # A=60, V=720
                shapely.Polygon(
                    [(19, -30), (28, -30), (28, -20), (19, -20)],
                ),  # A=90, V=2250
            ],
        }

        self.proj_crs = "EPSG:3347"
        self.data = gpd.GeoDataFrame(building_data, crs=self.proj_crs)
        self.buildings = Buildings(data=self.data, proj_crs=self.proj_crs)
        self.save_folder = Path(__file__).parent / "test_files" / "test_buildings_save"

    def test_create_size_flag(self) -> None:
        b = self.buildings
        buildings = b.create_size_flag(
            min_vol=500,
            max_vol=2000,
            flag_name="sfh",
            min_area=51,
            max_area=79,
        )

        # The column should be added
        self.assertTrue("sfh" in buildings.data)

        # Correct buildings are flagged
        self.assertFalse(buildings.data.iloc[0]["sfh"])  # Area 50 < 51
        self.assertFalse(buildings.data.iloc[1]["sfh"])  # Area 80 > 79
        self.assertTrue(buildings.data.iloc[2]["sfh"])
        self.assertFalse(buildings.data.iloc[3]["sfh"])  # Volume 2250 > 2000

    def test_calc_floors_no_breakpoints(self) -> None:
        buildings = self.buildings

        buildings = buildings.create_size_flag(
            min_vol=600,
            max_vol=2000,
            flag_name="sfh",
        )
        buildings = buildings.calc_floors(floor_height=5, type_col="sfh")

        # Should have a floors column
        self.assertTrue("floors" in buildings.data)

        # Non-sfh should have 0 floors
        self.assertEqual(buildings.data.iloc[0]["floors"], 0)
        self.assertEqual(buildings.data.iloc[3]["floors"], 0)

        # Building with height exactly equal to 4 * floor height should be four floors (not 5)
        self.assertEqual(buildings.data.iloc[1]["floors"], 4)

        # Building with height greater than 2 * floor height should be three floors
        self.assertEqual(buildings.data.iloc[2]["floors"], 3)

    def test_calc_floors_breakpoints(self) -> None:
        buildings = self.buildings

        buildings = buildings.calc_floors(
            floor_height=5,
            floor_breakpoints=[10, 20],
        )
        # Should have a floors column
        self.assertTrue("floors" in buildings.data)

        # Building with a height of -1 should have 0 floors
        self.assertEqual(buildings.data.iloc[0]["floors"], 0)

        self.assertEqual(buildings.data.iloc[1]["floors"], 2)
        self.assertEqual(buildings.data.iloc[2]["floors"], 2)
        self.assertEqual(buildings.data.iloc[3]["floors"], 3)

    def test_create_voronoi_polygons(self) -> None:
        buildings = self.buildings

        voronoi_polys = buildings.create_voronoi_polygons(
            boundary=None,
            flag_col=None,
            debuff=0.01,
        )

        # Return type should be a geoseries
        self.assertIsInstance(voronoi_polys, gpd.GeoSeries)
        self.assertIsInstance(voronoi_polys[0], shapely.Polygon)

        # lengths should be the same
        self.assertEqual(len(buildings.data), len(voronoi_polys))

    def test_create_voronoi_polygons_no_buildings(self):
        buildings = gpd.GeoDataFrame(
            data={"id": [pd.NA]},
            geometry=[pd.NA],
            crs="EPSG:4326",
        )
        buildings = Buildings(buildings, "EPSG:3347")

        voronoi_polys = buildings.create_voronoi_polygons(
            boundary=shapely.Polygon([(0, 0), (5, 0), (5, 10), (0, 10)]),
            flag_col=None,
        )

        self.assertIsInstance(voronoi_polys, gpd.GeoSeries)

        self.assertEqual(len(voronoi_polys), 1)
        self.assertEqual(sum(voronoi_polys.notna()), 0)

    def test_sjoin_addresses(self) -> None:
        # Setup
        ad_df = gpd.GeoDataFrame(
            {
                "address": [
                    "101 Main St",
                    "202 Second Ave",
                    "303 Third Blvd",
                    "404 Fourth Rd",
                    "505 Fifth Cres",
                ],
                "geometry": [
                    shapely.Point(0.5, 0.5),  # Inside building 0
                    shapely.Point(2, 2),  # Inside building 0
                    shapely.Point(-10, -9),  # 1m above building 1
                    shapely.Point(15, 16),  # 6m above building 22
                    shapely.Point(9.5, 2),  # 3.5m left of 22, 4.5 m right of 1
                ],
            },
            crs=self.proj_crs,
        )

        buildings = self.buildings

        buildings = buildings.sjoin_addresses(
            ad_df,
            join_nearest=True,
            max_distance=5,
        )

        output = buildings.data.copy()
        output.index = output["id"]  # for convenience

        # Building 0 should have the first two addresses, but not the fifth one
        self.assertIn(0, output.loc[0]["address_indices"])
        self.assertIn(1, output.loc[0]["address_indices"])
        self.assertNotIn(4, output.loc[0]["address_indices"])

        # Building 1 should have the third address
        self.assertIn(2, output.loc[1]["address_indices"])

        # Building 22 should have the fifth address
        self.assertIn(4, output.loc[22]["address_indices"])

        # Building 22 and 31 should have nothing
        self.assertEqual(len(output.loc[31]["address_indices"]), 0)

        # Data should be unchanged
        df = buildings.data[self.buildings.data.columns]
        assert_geodataframe_equal(df, self.buildings.data)


class TestShrinkBuildings(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

        # Sample geometries for testing
        self.geoms = gpd.GeoSeries(
            [
                shapely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),  # Simple square
                shapely.Polygon([(4, 0), (6, 0), (6, 2), (4, 2)]),  # Simple square
                shapely.Polygon(
                    [
                        (0, 3),
                        (3, 3),
                        (3, 8),
                        (0, 8),
                        (0, 6),
                        (2, 6),
                        (2, 5),
                        (0, 5),
                    ],
                ),  # Backwards C shape with width 2 on vertical. Multi poly when shrunk by 0.5
            ],
        )

    def test_shrink_buildings(self):
        debuff_size = 0.1
        result = _shrink_buildings(self.geoms, debuff_size)

        # Should be the same number of geoms
        self.assertTrue(len(result), len(self.geoms))

        # Geoms should be smaller
        for g, res in zip(self.geoms, result, strict=True):
            self.assertTrue(res.area < g.area)

    def test_shrink_buildings_with_fix_multi(self):
        debuff_size = 0.5
        result = _shrink_buildings(self.geoms, debuff_size, fix_multi=True)

        # Should be one more geom
        self.assertEqual(len(result), (len(self.geoms)))

        # Geoms
        self.assertLess(result[0].area, self.geoms[0].area)
        self.assertLess(result[1].area, self.geoms[1].area)
        self.assertTrue(result[2].geom_type == "Polygon")
        self.assertEqual(self.geoms[2].area, result[2].area)

        for g, res in zip(self.geoms, result, strict=True):
            self.assertTrue(res.area <= g.area)

    def test_shrink_buildings_without_fix_multi(self):
        debuff_size = 0.6
        result = _shrink_buildings(self.geoms, debuff_size, fix_multi=False)

        self.assertTrue(result[2].geom_type == "MultiPolygon")

    def test_no_shrink_when_debuff_zero(self):
        # Test no change when debuff size is zero
        debuff_size = 0.0
        result = _shrink_buildings(self.geoms, debuff_size)

        assert_geoseries_equal(self.geoms, result)


class TestDissolveOverlaps(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

    def test_overlapping_polygons_dissolve(self):
        # Create two overlapping polygons
        poly1 = shapely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = shapely.Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        data = {"geometry": [poly1, poly2], "id": [0, 1], "value": ["a", "b"]}
        gdf = gpd.GeoDataFrame(data)

        result = _dissolve_overlaps(gdf)

        self.assertIn("id", result)
        self.assertIn("value", result)
        self.assertNotIn("group", result)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["value"], "a")
        self.assertEqual(result.iloc[0]["id"], 0)

    def test_touching_polygons_remain_separate(self):
        # Create two touching polygons
        poly1 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = shapely.Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
        data = {"geometry": [poly1, poly2], "id": [1, 2], "value": ["a", "b"]}
        gdf = gpd.GeoDataFrame(data)

        result = _dissolve_overlaps(gdf)

        self.assertEqual(len(result), 2)
        assert_geodataframe_equal(gdf, result)

    def test_identical_polygons(self):
        poly1 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        poly2 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        poly3 = shapely.Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])

        data = {
            "geometry": [poly1, poly2, poly3],
            "id": [0, 1, 2],
            "value": ["a", "b", "c"],
        }
        gdf = gpd.GeoDataFrame(data)

        expected = gpd.GeoDataFrame(
            {
                "geometry": [poly1, poly3],
                "id": [0, 2],
                "value": ["a", "c"],
            },
        )

        result = _dissolve_overlaps(gdf)

        assert_geodataframe_equal(result, expected, normalize=True, check_like=True)


if __name__ == "__main__":
    unittest.main()
