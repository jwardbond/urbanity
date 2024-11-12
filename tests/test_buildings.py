import os
import sys
import unittest
import warnings
from pathlib import Path


import shapely
import geopandas as gpd

from urbanity import Buildings
from geopandas.testing import assert_geodataframe_equal


os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]), "Library", "share", "gdal"
)  # HACK GDAL warning suppression


class TestBuildings(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter(
            "ignore", category=DeprecationWarning
        )  # HACK geopandas warning suppression

        # Create a mock region

        building_data = {
            "id": [0, 1, 2, 3],
            "height": [10.0, 20.0, 12.0, 25.0],
            "geometry": [
                shapely.Polygon([(0, 0), (5, 0), (5, 10), (0, 10)]),  # A=50, V=500
                shapely.Polygon([(0, 0), (8, 0), (8, 10), (0, 10)]),  # A=80, V=1600
                shapely.Polygon([(0, 0), (6, 0), (6, 10), (0, 10)]),  # A=60, V=720
                shapely.Polygon([(0, 0), (9, 0), (9, 10), (0, 10)]),  # A=90, V=2250
            ],
        }

        self.proj_crs = "EPSG:3347"
        self.data = gpd.GeoDataFrame(building_data, crs=self.proj_crs)
        self.buildings = Buildings(data=self.data, proj_crs=self.proj_crs)

    def test_init(self) -> None:
        buildings = Buildings(data=self.data, proj_crs=self.proj_crs)

        self.assertTrue("area" in buildings.data)
        self.assertTrue("id" in buildings.data)

        # Area calcs should be correct
        self.assertAlmostEqual(90, buildings.data.iloc[3]["area"])

    def test_read_geojson(self) -> None:
        saved = Buildings.read_geojson(
            Path("./tests/test_files/test_files_mock_buildings.geojson"),
            proj_crs=self.proj_crs,
        )

        new = Buildings(data=self.data, proj_crs=self.proj_crs)

        # Saved and new dataframe should be the same (check to make sure you haven't changed the new dataframe)
        assert_geodataframe_equal(saved.data, new.data)

        # They should both have the same projected crs system
        self.assertEqual(saved.proj_crs, new.proj_crs)

    def test_create_volume_flag(self) -> None:
        b = self.buildings
        buildings = b.create_volume_flag(min_vol=600, max_vol=2000, flag_name="sfh")

        # The column should be added
        self.assertTrue("sfh" in buildings.data)

        # Correct buildings are flagged
        self.assertFalse(buildings.data.iloc[0]["sfh"])
        self.assertTrue(buildings.data.iloc[1]["sfh"])
        self.assertTrue(buildings.data.iloc[2]["sfh"])
        self.assertFalse(buildings.data.iloc[3]["sfh"])

    def test_calc_floors_no_breakpoints(self) -> None:
        buildings = self.buildings

        buildings = buildings.create_volume_flag(
            min_vol=600, max_vol=2000, flag_name="sfh"
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

        # Building with height on breakpoint should not have an extra floor
        self.assertEqual(buildings.data.iloc[0]["floors"], 1)
        self.assertEqual(buildings.data.iloc[1]["floors"], 2)

        # Building with height greater than 10m breakpoint should be 2 floors
        self.assertEqual(buildings.data.iloc[2]["floors"], 2)
        self.assertEqual(buildings.data.iloc[3]["floors"], 3)
