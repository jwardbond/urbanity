import copy
import os
import shutil
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import shapely
from geopandas.testing import assert_geodataframe_equal

from urbanity import Buildings

os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]),
    "Library",
    "share",
    "gdal",
)  # HACK GDAL warning suppression


class TestBuildings(unittest.TestCase):
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

    def test_init(self) -> None:
        buildings = Buildings(data=self.data, proj_crs=self.proj_crs)

        self.assertTrue("area" in buildings.data)
        self.assertTrue("id" in buildings.data)

        # Area calcs should be correct
        self.assertAlmostEqual(90, buildings.data.iloc[3]["area"], places=5)

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

    def test_get_voronoi_plots(self) -> None:
        buildings = self.buildings

        voronoi_polys = buildings.create_voronoi_plots(
            boundary=None,
            min_building_footprint=0,
            shrink=False,
            building_rep="mrr",
        )

        # Return type should be list of tuples
        self.assertIsInstance(voronoi_polys, list)
        self.assertIsInstance(voronoi_polys[0], tuple)

        # lengths should be the same
        self.assertEqual(len(buildings.data), len(voronoi_polys))

        # building ids should still be in the voronoi polygons data
        self.assertTrue(
            all(i[1] in buildings.data["id"].to_list() for i in voronoi_polys),
        )

    @patch("pathlib.Path.mkdir")
    @patch("utils.save_geodf")
    def test_save_only_one_geom(self, mock_save, mock_mkdir) -> None:  # noqa: ANN001, ARG002
        buildings = copy.deepcopy(self.buildings)

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

    @patch("pathlib.Path.mkdir")
    @patch("utils.save_geodf")
    def test_save_multiple_geom(self, mock_save, mock_mkdir) -> None:  # noqa: ANN001, ARG002
        buildings = copy.deepcopy(self.buildings)
        buildings.data["extra_one"] = buildings.data["geometry"]
        buildings.data["extra_two"] = buildings.data["extra_one"]

        buildings.save(self.save_folder)

        # Utils save should only be called once
        self.assertEqual(mock_save.call_count, 3)

        # FIRST CALL
        args, _ = mock_save.call_args_list[0]
        gdf, path = args
        expected_path = self.save_folder / "extra_one"

        # Path should be correct
        self.assertEqual(path, expected_path)

        # Dataframe should have only an id and geometry column
        self.assertTrue("id" in gdf)
        self.assertTrue("geometry" in gdf)
        self.assertEqual(len(gdf.columns), 2)

        # Dataframe should have only one column containing geometries
        self.assertEqual(
            sum(
                isinstance(gdf[c].iloc[0], shapely.geometry.base.BaseGeometry)
                for c in gdf.columns
            ),
            1,
        )

        # SECOND CALL
        args, _ = mock_save.call_args_list[1]
        gdf, path = args
        expected_path = self.save_folder / "extra_two"

        # Path should be correct
        self.assertEqual(path, expected_path)

        # Dataframe should have only an id and geometry column
        self.assertTrue("id" in gdf)
        self.assertTrue("geometry" in gdf)
        self.assertEqual(len(gdf.columns), 2)

        # Dataframe should have only one column containing geometries
        self.assertEqual(
            sum(
                isinstance(gdf[c].iloc[0], shapely.geometry.base.BaseGeometry)
                for c in gdf.columns
            ),
            1,
        )

        # THIRD CALL
        args, _ = mock_save.call_args_list[2]
        gdf, path = args
        expected_path = self.save_folder / "buildings"

        # Path should be correct
        self.assertEqual(path, expected_path)

        # Dataframe should have all the original columns except for extra_one and extra_two
        self.assertTrue(all(c in gdf for c in self.data.columns))

        # Dataframe should have only one column containing geometries
        self.assertEqual(
            sum(
                isinstance(gdf[c].iloc[0], shapely.geometry.base.BaseGeometry)
                for c in gdf.columns
            ),
            1,
        )

    def test_save_and_load(self) -> None:
        buildings = copy.deepcopy(self.buildings)
        buildings.data["extra_one"] = buildings.data["geometry"]
        buildings.data["extra_two"] = buildings.data["extra_one"]

        if self.save_folder.exists():
            shutil.rmtree(self.save_folder)

        buildings.save(self.save_folder)
        loaded = Buildings.load(self.save_folder, proj_crs="EPSG:3347")

        # Files should be created
        self.assertTrue(Path(self.save_folder / "buildings.gpkg").exists())
        self.assertTrue(Path(self.save_folder / "extra_one.gpkg").exists())
        self.assertTrue(Path(self.save_folder / "extra_two.gpkg").exists())

        # Data should be unchanged
        assert_geodataframe_equal(loaded.data, buildings.data, check_like=True)
