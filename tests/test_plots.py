import copy
import os
import shutil
import sys
import unittest
import warnings
from pathlib import Path

import geopandas as gpd
import shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal

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


class TestCreatePredicateFlag(unittest.TestCase):
    def setUp(self):
        # Sample plots
        plot_geom = [shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        plot_data = gpd.GeoDataFrame(
            {"id": [1], "geometry": plot_geom},
            crs="EPSG:4326",
        )
        self.plots = Plots(data=plot_data, proj_crs="EPSG:3347")

        # Create sample polygon data
        poly_geom = [
            shapely.Polygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]),
        ]
        self.poly_data = gpd.GeoDataFrame({"geometry": poly_geom}, crs="EPSG:4326")

    def test_predicate_flag_matching_crs(self):
        result = self.plots.create_predicate_flag(
            self.poly_data,
            predicate="contains",
            flag_name="test_flag",
        )
        self.assertTrue("test_flag" in result.data.columns)
        self.assertTrue(result.data["test_flag"].iloc[0])

    def test_predicate_flag_different_crs(self):
        # Change polygon CRS
        self.poly_data.crs = "EPSG:3857"
        with self.assertWarns(Warning):
            result = self.plots.create_predicate_flag(
                self.poly_data,
                predicate="contains",
                flag_name="test_flag",
            )
        self.assertTrue("test_flag" in result.data.columns)

    def test_predicate_flag_duplicate_name(self):
        # Add flag column first
        self.plots.data["pred_flag"] = True
        with self.assertRaises(ValueError):
            self.plots.create_predicate_flag(self.poly_data, "contains")


class TestSjoinMostIntersecting(unittest.TestCase):
    def setUp(self):
        # Create test plot data
        plot_data = gpd.GeoDataFrame(
            {
                "id": [1, 2],
                "geometry": [
                    shapely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                    shapely.Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
                ],
            },
            crs="EPSG:4326",
        )
        self.plots = Plots(plot_data, "EPSG:3857")

        # Create test building data with multiple overlapping buildings
        self.buildings = gpd.GeoDataFrame(
            {
                "building_type": ["residential", "commercial", "school"],
                "height": [10, 20, 15],
                "geometry": [
                    # Large overlap with plot 1
                    shapely.Polygon([(0.2, 0.2), (1.8, 0.2), (1.8, 1.8), (0.2, 1.8)]),
                    # Small overlap with plot 1
                    shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    # Overlap with plot 2
                    shapely.Polygon([(3.2, 3.2), (4.8, 3.2), (4.8, 4.8), (3.2, 4.8)]),
                ],
            },
            crs="EPSG:4326",
        )

    def test_multiple_buildings_largest_intersection(self):
        result = self.plots.sjoin_most_intersecting(
            self.buildings,
            ["building_type", "height"],
        )

        # Plot 1 should get data from first building (larger overlap)
        self.assertEqual(result.data.loc[0, "building_type"], "residential")
        self.assertEqual(result.data.loc[0, "height"], 10)

        # Plot 2 should get data from third building
        self.assertEqual(result.data.loc[1, "building_type"], "school")
        self.assertEqual(result.data.loc[1, "height"], 15)

    def test_preserves_original_crs_and_geometry(self):
        result = self.plots.sjoin_most_intersecting(self.buildings, ["building_type"])

        assert_geoseries_equal(result.data.geometry, self.plots.data.geometry)
        self.assertEqual(result.data.crs, self.plots.data.crs)

    def test_raises_error_on_duplicate_columns(self):
        # Add column that exists in buildings
        self.plots.data["building_type"] = ["test1", "test2"]
        plots = Plots(self.plots.data, "EPSG:3857")

        with self.assertRaises(ValueError):
            plots.sjoin_most_intersecting(self.buildings, ["building_type"])


class TestUpdate(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.geom1 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.geom2 = shapely.Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])

        self.base_data = gpd.GeoDataFrame(
            {
                "id": [1, 2],
                "geometry": [self.geom1, self.geom2],
                "area": [1.0, 1.0],
                "value1": [10, 20],
            },
            crs="EPSG:4326",
        )

        self.other_data = gpd.GeoDataFrame(
            {
                "id": [1, 2],
                "geometry": [self.geom1, self.geom2],
                "area": [1.0, 1.0],
                "value2": [30, 40],
            },
            crs="EPSG:4326",
        )

        self.base_plots = Plots(self.base_data, "EPSG:4326")
        self.other_plots = Plots(self.other_data, "EPSG:4326")

    def test_update_passes_parameters_correctly(self):
        result = self.base_plots.update(
            self.other_plots,
            cols=["value2"],
            overwrite=True,
        )
        self.assertIn("value2", result.data.columns)
        self.assertEqual(result.data["value2"].tolist(), [30, 40])

    def test_update_preserves_required_columns(self):
        result = self.base_plots.update(
            self.other_plots, cols=["id", "geometry", "area", "value2"], overwrite=True
        )
        self.assertEqual(result.data["id"].tolist(), self.base_data["id"].tolist())
        self.assertEqual(
            result.data.geometry.tolist(), self.base_data.geometry.tolist()
        )
        self.assertEqual(result.data["area"].tolist(), self.base_data["area"].tolist())

    def test_update_raises_on_overwrite_without_permission(self):
        other_data = self.other_data.copy()
        other_data["value1"] = [50, 60]
        other_plots = Plots(other_data, "EPSG:4326")

        with self.assertRaises(ValueError):
            self.base_plots.update(other_plots, cols=["value1"], overwrite=False)


class TestUpdateFromGdf(unittest.TestCase):
    def setUp(self):
        # Create base Plots object
        self.base_geom = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.base_data = gpd.GeoDataFrame(
            {
                "id": [1],
                "geometry": [self.base_geom],
                "area": [1.0],
                "existing_col": ["old_value"],
            },
            crs="EPSG:4326",
        )
        self.base_plots = Plots(self.base_data, proj_crs="EPSG:3347")

    def test_update_from_gdf_basic_merge(self):
        # Create other GDF with new column
        other_data = gpd.GeoDataFrame({"id": [1], "new_col": ["new_value"]})

        updated = self.base_plots.update_from_gdf(other_data, cols=["new_col"])

        row = updated.data[updated.data["id"] == 1].iloc[0]
        self.assertEqual(row["new_col"], "new_value")
        self.assertEqual(row["existing_col"], "old_value")
        self.assertEqual(row["area"], 1.0)
        self.assertEqual(row["geometry"], self.base_geom)

    def test_update_from_gdf_with_overwrite(self):
        # Create other GDF with column that exists in base
        other_data = gpd.GeoDataFrame({"id": [1], "existing_col": ["new_value"]})

        updated = self.base_plots.update_from_gdf(
            other_data, cols=["existing_col"], overwrite=True
        )

        row = updated.data[updated.data["id"] == 1].iloc[0]
        self.assertEqual(row["existing_col"], "new_value")
        self.assertEqual(row["area"], 1.0)
        self.assertEqual(row["geometry"], self.base_geom)

    def test_update_from_gdf_missing_id_column(self):
        other_data = gpd.GeoDataFrame({"new_col": ["new_value"]})

        with self.assertRaises(KeyError):
            self.base_plots.update_from_gdf(other_data, cols=["new_col"])

    def test_update_from_gdf_overwrite_protection(self):
        other_data = gpd.GeoDataFrame({"id": [1], "existing_col": ["new_value"]})

        with self.assertRaises(ValueError):
            self.base_plots.update_from_gdf(
                other_data, cols=["existing_col"], overwrite=False
            )


if __name__ == "__main__":
    unittest.main()
