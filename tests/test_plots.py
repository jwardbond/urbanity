import copy
import os
import shutil
import sys
import unittest
import warnings
from pathlib import Path

import geopandas as gpd
import shapely
from geopandas.testing import assert_geodataframe_equal

from urbanity.plots import Plots

os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]),
    "Library",
    "share",
    "gdal",
)  # HACK GDAL warning suppression


class TestPlotsInit(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppress

        # Create sample geometries
        # Simple rectangular plots
        geometries = [
            shapely.box(0, 0, 10, 10),  # 100 sq units
            shapely.box(20, 0, 25, 10),  # 50 sq units
            shapely.box(30, 0, 35, 5),  # 25 sq units
            shapely.box(40, 0, 49, 10),  # 90 sq units
        ]

        plot_data = {"id": [1, 3, 17, 21], "geometry": geometries}

        # Create GeoDataFrame with the test geometries
        self.data = gpd.GeoDataFrame(
            {"geometry": geometries, "id": range(1, len(geometries) + 1)},
            crs="EPSG:4326",  # WGS 84
        )

        # Define projection CRS for testing
        self.proj_crs = "EPSG:3347"
        self.data = gpd.GeoDataFrame(plot_data, crs=self.proj_crs)
        self.plots = Plots(data=self.data, proj_crs=self.proj_crs)
        self.save_folder = Path(__file__).parent / "test_files" / "test_plots_save"

    def test_init(self) -> None:
        plots = Plots(data=self.data, proj_crs=self.proj_crs)

        self.assertTrue("area" in plots.data)
        self.assertTrue("id" in plots.data)

        # Area calcs should be correct
        self.assertAlmostEqual(90, plots.data.iloc[3]["area"], places=5)

    def test_save_and_load(self) -> None:
        plots = copy.deepcopy(self.plots)

        if self.save_folder.exists():
            shutil.rmtree(self.save_folder)

        plots.save(self.save_folder)
        loaded = Plots.load(self.save_folder, proj_crs="EPSG:3347")

        # Files should be created
        self.assertTrue(Path(self.save_folder / "plots.parquet").exists())

        # Data should be unchanged
        assert_geodataframe_equal(loaded.data, plots.data, check_like=True)


class TestSubtractPolygons(unittest.TestCase):
    def setUp(self):
        # Create sample plot data
        plot_geom = shapely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])  # A=4.0
        plot_data = gpd.GeoDataFrame({"geometry": [plot_geom]}, crs="EPSG:3347")
        self.plots = Plots(data=plot_data, proj_crs="EPSG:3347")

    def test_subtract_polygons_with_matching_crs(self):
        # Create polygon to subtract
        subtract_geom = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])  # A=1.0
        to_subtract = gpd.GeoDataFrame({"geometry": [subtract_geom]}, crs="EPSG:3347")
        to_subtract = to_subtract.to_crs("EPSG:4326")

        result = self.plots.subtract_polygons(to_subtract)

        # Check result has correct area
        expected_area = 3.0  # Original area (4.0) - subtracted area (1.0)
        self.assertAlmostEqual(result.data["area"].sum(), expected_area)

    def test_subtract_polygons_with_different_crs(self):
        subtract_geom = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])  # A=1.0
        to_subtract = gpd.GeoDataFrame({"geometry": [subtract_geom]}, crs="EPSG:3347")

        with self.assertWarns(UserWarning):
            result = self.plots.subtract_polygons(to_subtract)

        self.assertEqual(result.data.crs, "EPSG:4326")


class TestCreateCircleFitFlag(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppress

        # Create test polygons
        square = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])  # can fit rad 5
        small_triangle = shapely.Polygon([(0, 0), (1, 0), (0, 1)])  # can fit
        point = shapely.Point(0, 0)

        # Define coordinates for the non-convex geometry
        large_square = shapely.Polygon(
            [(0, 0), (10, 0), (10, 11), (0, 10), (0, 0)],
        )  # A (slightly skewed but >) 10x10 square
        small_square = shapely.Polygon(
            [(7, 1), (9, 1), (9, 3), (7, 3), (7, 1)],
        )  # 2x2 square
        connector = shapely.Polygon(
            [(5, 2), (7, 2), (7, 3), (5, 3), (5, 2)],
        )  # Connecting rectangle
        non_convex = shapely.ops.unary_union([large_square, small_square, connector])

        # Create geodataframe with test geometries
        self.test_data = gpd.GeoDataFrame(
            geometry=[square, small_triangle, non_convex, point],
            crs="EPSG:3347",
        )
        self.plots = Plots(data=self.test_data, proj_crs="EPSG:3347")

    def test_create_circle_fit_flag_positive_cases(self):
        radius = 4.0
        tolerance = 0.001
        result = self.plots.create_circle_fit_flag(
            radius=radius,
            tolerance=tolerance,
            flag_name="can_fit",
        )

        # Square should fit circle with radius 4
        self.assertTrue(result.data.loc[0]["can_fit"])

        # Concave polygon should fit circle with radius 4
        self.assertTrue(result.data.loc[2]["can_fit"])

    def test_create_circle_fit_flag_negative_cases(self):
        radius = 4.0
        tolerance = 0.1
        result = self.plots.create_circle_fit_flag(
            radius=radius,
            tolerance=tolerance,
            flag_name="can_fit",
        )

        # Small triangle shouldn't fit circle with radius 4
        self.assertFalse(result.data.loc[1]["can_fit"])

        # Point geometry shouldn't fit any circle
        self.assertFalse(result.data.loc[3]["can_fit"])

    def test_raises_radius_error(self):
        self.assertRaises(ValueError, self.plots.create_circle_fit_flag, radius=0)


if __name__ == "__main__":
    unittest.main()
