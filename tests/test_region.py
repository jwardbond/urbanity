import copy
import os
import sys
import unittest
import warnings
from pathlib import Path

import geopandas as gpd
import shapely
from geopandas.testing import assert_geodataframe_equal

import utils
from urbanity import Buildings, Region

os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]),
    "Library",
    "share",
    "gdal",
)  # HACK GDAL warning suppression


class TestRegion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

        # Set output path and get rid of existing files
        networkpath = Path("./tests/test_files/test_files_road_network.geojson")
        cls.network = utils.load_geojson(networkpath)

        cls.proj_crs = "EPSG:3347"

    def test_init(self):
        network = self.network
        grid_size = 1024
        min_area = 10000
        width_thres = 20
        clust_width = 25
        point_precision = 2

        generated = Region.build_from_network(
            network=network,
            proj_crs=self.proj_crs,
            grid_size=grid_size,
            min_area=min_area,
            # max_area=40000,  # If you uncomment the two dataframes will not be equal
            width_thres=width_thres,
            clust_width=clust_width,
            point_precision=point_precision,
        )

        loaded = Region.load_from_files(
            segments=Path("./tests/test_files/test_files_segments.geojson"),
            proj_crs=self.proj_crs,
            road_network=Path(
                "./tests/test_files/test_files_road_network.geojson",
            ),
            buildings=Path("./tests/test_files/test_files_osm_buildings.geojson"),
        )

        # Test coordinate systems
        self.assertTrue("EPSG:4326", generated.segments.crs)
        self.assertTrue("EPSG:4326", loaded.segments.crs)

        # Segments and buildings should have area
        self.assertTrue("area" in loaded.buildings.data)
        self.assertTrue("area" in loaded.segments)
        self.assertTrue("area" in generated.segments)

        # Segments should all have an id
        self.assertTrue("id" in loaded.segments)
        self.assertTrue("id" in generated.segments)

        # Test that the segmentation generation is (still) running correctly
        assert_geodataframe_equal(generated.segments, loaded.segments)

    def test__split_polygon(self):
        side_length = 1000

        # Wide polygons
        wide_poly = shapely.Polygon(
            (
                (0, 0),
                (side_length * 8, 0),
                (side_length * 8, side_length * 4),
                (0, side_length * 4),
            ),
        )

        wide_split_left = shapely.Polygon(
            (
                (0, 0),
                (side_length * 4, 0),
                (side_length * 4, side_length * 4),
                (0, side_length * 4),
            ),
        )
        wide_split_right = shapely.Polygon(
            (
                (side_length * 4, 0),
                (side_length * 8, 0),
                (side_length * 8, side_length * 4),
                (side_length * 4, side_length * 4),
            ),
        )

        # Tall polygons
        tall_poly = shapely.affinity.rotate(wide_poly, 90, origin=(0, 0))
        tall_split_upper = shapely.affinity.rotate(wide_split_right, 90, origin=(0, 0))
        tall_split_lower = shapely.affinity.rotate(wide_split_left, 90, origin=(0, 0))

        split_wide = Region._split_polygon(wide_poly)
        split_tall = Region._split_polygon(tall_poly)

        # Should only divide the polygon once
        self.assertEqual(len(split_wide), 2)
        self.assertEqual(len(split_tall), 2)

        # Should divide a wide poly in half with a vertical line in the middle
        self.assertTrue(shapely.equals(wide_split_left, split_wide[0]))
        self.assertTrue(shapely.equals(wide_split_right, split_wide[1]))

        # Should divide a tall polygon in half with a horizontal line in the center
        self.assertTrue(shapely.equals(tall_split_lower, split_tall[0]))
        self.assertTrue(shapely.equals(tall_split_upper, split_tall[1]))

    def test__subdivide_segments(self):
        side_length = 10
        max_area = 100

        # Wide polygons
        poly = shapely.Polygon(
            (
                (0, 0),
                (side_length * 8, 0),
                (side_length * 8, side_length * 4),
                (0, side_length * 4),
            ),
        )

        segments = gpd.GeoDataFrame(geometry=[poly, poly], crs=self.proj_crs)
        segments = Region._subdivide_segments(segments, max_area=max_area)

        # There should be 32 segments
        self.assertEqual(len(segments), 64)

        # No segment should have an area bigger than max_area
        self.assertTrue((segments["geometry"].area <= max_area).all())


class TestRegionFeatureMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

        # Create a mock region
        square1 = shapely.Polygon([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
        square2 = shapely.Polygon(
            [(100, 0), (200, 0), (200, 100), (100, 100), (100, 0)],
        )
        proj_crs = "EPSG:3347"
        segments = gpd.GeoDataFrame({"geometry": [square1, square2]}, crs=proj_crs)

        segments = segments.to_crs("EPSG:4326")
        cls.region = Region(segments, proj_crs)

        # Create some smaller mock polygons to overlay
        circle1 = shapely.Point(150, 80).buffer(20)
        circle2 = shapely.Point(50, 20).buffer(15)
        circle3 = shapely.Point(150, 40).buffer(10)
        circle4 = shapely.Point(40, 80).buffer(20)

        buildings = gpd.GeoDataFrame(
            {"geometry": [circle1, circle2, circle3, circle4]},
            crs=proj_crs,
        )
        buildings["height"] = buildings["geometry"].area
        buildings["area"] = buildings["geometry"].area

        buildings = buildings.to_crs("EPSG:4326")
        cls.buildings = buildings

    def test_subtract_polygons(self) -> None:
        region = copy.deepcopy(self.region)
        polygons = copy.deepcopy(self.buildings)

        output = region.subtract_polygons(polygons)

        # region should be unchanged
        self.assertTrue(region == self.region)

        # output should be different
        self.assertFalse(output == region)

        # output should have same crs
        self.assertEqual(output.proj_crs, region.proj_crs)

        # output should have WSG84 CRS
        self.assertEqual(output.segments.crs, "EPSG:4326")

        # output should have a smaller total area
        self.assertLess(
            output.segments.to_crs(output.proj_crs).area.sum(),
            region.segments.to_crs(region.proj_crs).area.sum(),
        )
        # self.assertAlmostEqual(
        #     output.segments.to_crs(output.proj_crs).area.sum(),
        #     region.segments.to_crs(region.proj_crs).area.sum()
        #     - polygons.to_crs(region.proj_crs).area.sum(),
        # )

    def test_agg_features(self) -> None:
        region = copy.deepcopy(self.region)
        polygons = copy.deepcopy(self.buildings)
        output = region.agg_features(
            polygons,
            feature_name="height",
            how="mean",
            fillnan=0,
        )

        # region should be unchanged
        self.assertTrue(region == self.region)

        # output should be different
        self.assertFalse(output == region)

        # output should have same crs
        self.assertEqual(output.proj_crs, region.proj_crs)

        # output should have WSG84 CRS
        self.assertEqual(output.segments.crs, "EPSG:4326")

        # output should have the same total area
        self.assertAlmostEqual(
            output.segments.to_crs(output.proj_crs).area.sum(),
            region.segments.to_crs(region.proj_crs).area.sum(),
        )

    def test_disagg_features(self) -> None:
        region = copy.deepcopy(self.region)

        circle1 = shapely.Point(100, 80).buffer(20)
        circle2 = shapely.Point(70, 20).buffer(15)

        polygons = gpd.GeoDataFrame(
            {"geometry": [circle1, circle2]},
            crs=region.proj_crs,
        )
        polygons["pop"] = polygons["geometry"].area

        # run disaggregator <-sounds made-up
        output = region.disagg_features(polygons, "pop", how="area")

        # region should be unchanged
        self.assertTrue(region == self.region)

        # output should be different
        self.assertFalse(output == region)

        # output should have same crs
        self.assertEqual(output.proj_crs, region.proj_crs)

        # output should have WSG84 CRS
        self.assertEqual(output.segments.crs, "EPSG:4326")

        # output should have the same total area
        self.assertAlmostEqual(
            output.segments.to_crs(output.proj_crs).area.sum(),
            region.segments.to_crs(region.proj_crs).area.sum(),
            4,
        )

        # output should have a new feature
        self.assertTrue("pop" in output.segments)

        # the population in the leftmost region should be circle2 and half of circle1
        self.assertAlmostEqual(
            output.segments[output.segments["id"] == 0]["pop"].sum(),
            circle1.area / 2 + circle2.area,
        )

        self.assertAlmostEqual(
            output.segments[output.segments["id"] == 1]["pop"].sum(),
            circle1.area / 2,
        )

    def test_tiling(self) -> None:
        # TODO
        pass


class TestRegionMethodsWithBuildings(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

        self.proj_crs = "EPSG:3347"

        # Create two large squares
        seg1 = shapely.Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
        seg2 = shapely.Polygon([(20, 0), (20, 10), (30, 10), (30, 0), (20, 0)])
        seg3 = shapely.Polygon([(40, 0), (40, 10), (50, 10), (50, 0), (40, 0)])

        segments = gpd.GeoDataFrame(
            {"id": [0, 1, 2]},
            geometry=[seg1, seg2, seg3],
            crs=self.proj_crs,
        )

        segments = segments.to_crs("EPSG:4326")
        self.region = Region(segments, self.proj_crs)

        # Create four smaller polygons, two in each large square
        # In segment 1
        building0 = shapely.Polygon([(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)])
        building1 = shapely.Polygon([(4, 1), (4, 3), (6, 3), (6, 1), (4, 1)])
        building2 = shapely.Polygon([(7, 1), (7, 3), (9, 3), (9, 1), (7, 1)])
        building3 = shapely.Polygon([(1, 6), (1, 8), (3, 8), (3, 6), (1, 6)])
        building4 = shapely.Polygon([(6, 6), (6, 8), (8, 8), (8, 6), (6, 6)])
        building5 = shapely.Polygon([(9, 6), (9, 8), (11, 8), (11, 6), (9, 6)])

        # in segment 2
        building6 = shapely.Polygon([(21, 1), (21, 3), (23, 3), (23, 1), (21, 1)])
        building7 = shapely.Polygon([(24, 1), (24, 3), (26, 3), (26, 1), (24, 1)])
        building8 = shapely.Polygon([(21, 6), (21, 8), (23, 8), (23, 6), (21, 6)])
        building9 = shapely.Polygon([(24, 6), (24, 8), (26, 8), (26, 6), (24, 6)])

        # in segment 3
        building10 = shapely.Polygon([(41, 1), (41, 3), (43, 3), (43, 1), (41, 1)])
        building11 = shapely.Polygon([(46, 1), (46, 3), (48, 3), (48, 1), (46, 1)])

        # 6 sfh in first segment, 4 in second, 2 in third
        buildings = gpd.GeoDataFrame(
            {"id": list(range(12)), "sfh": [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1]},
            geometry=[
                building0,
                building1,
                building2,
                building3,
                building4,
                building5,
                building6,
                building7,
                building8,
                building9,
                building10,
                building11,
            ],
            crs=self.proj_crs,
        )
        buildings = buildings.to_crs("EPSG:4326")
        buildings = Buildings(buildings, self.proj_crs)

        self.region.buildings = buildings

    def test_flag_segments_by_buildings(self):
        region = self.region
        old_segments = region.segments.copy()

        region = region.flag_segments_by_buildings(
            flag_name="sfh",
            threshold_pct=0.7,
            threshold_num=3,
            building_flag="sfh",
        )

        segments = region.segments

        # Columns should be the same + one new column
        self.assertTrue("sfh" in segments)
        self.assertTrue(all(col in segments for col in old_segments.columns))

        # Data should be unchanged without the new column
        assert_geodataframe_equal(segments.drop(columns=["sfh"]), old_segments)

        # Segment 1 should not be flagged (threshold pct is 4/6 <= 0.7)
        self.assertFalse(segments.iloc[0]["sfh"])

        # Segment 2 should be flagged (threshold pct is 3/4 >= 0.7)
        self.assertTrue(segments.iloc[1]["sfh"])

        # Segment 3 should not be flagged (threshold num is 2 <= 3)
        self.assertFalse(segments.iloc[2]["sfh"])

    def test_add_pseudo_plots(self):
        """Test basic functionality of add_pseudo_plots."""
        region = copy.deepcopy(self.region)

        region = region.flag_segments_by_buildings(
            flag_name="sfh",
            threshold_pct=0.7,
            threshold_num=3,
            building_flag="sfh",
        )
        result = region.add_pseudo_plots(segment_flag="sfh")

        # Building ids 6,7,8,9 should have pseudo plots, the rest should not
        filtered = result.buildings.data[result.buildings.data["pseudo_plot"].notna()]
        self.assertTrue(all(i in filtered["id"] for i in [6, 7, 8, 9]))

        # ###
        # from matplotlib import pyplot as plt
        # import numpy as np

        # fig, ax = plt.subplots(figsize=(10, 10))

        # # Pseudo plots
        # pplots = gpd.GeoDataFrame(
        #     geometry=result.buildings.data["pseudo_plot"],
        #     crs=result.buildings.data.crs,
        # )
        # colors2 = [tuple(np.random.rand(3)) for _ in range(len(pplots))]
        # pplots.plot(ax=ax, color=colors2)

        # # Buildings
        # result.buildings.data.plot(ax=ax, color="grey")

        # plt.show()
        # ###

        # Buildings should be within voronoi polys
        self.assertTrue(
            all(
                shapely.within(r["geometry"], r["pseudo_plot"].buffer(1e-9))
                for _, r in filtered.iterrows()
            ),
        )

        # BASIC FUNCTIONALITY TESTS
        # Should return an instance of Region
        self.assertIsInstance(result, Region)

        # Building data should have a "voronoi_geometry" column
        self.assertTrue("pseudo_plot" in result.buildings.data.columns)

        # Length of building and segment data should be unchanged
        self.assertEqual(len(result.buildings.data), len(self.region.buildings.data))
        self.assertEqual(len(result.segments), len(self.region.segments))

        # TODO might be able to abstract these tests
        # Projected CRS property should be unchanged
        self.assertEqual(result.buildings.proj_crs, self.region.buildings.proj_crs)
        self.assertEqual(self.region.buildings.proj_crs, result.proj_crs)
        self.assertEqual(result.proj_crs, self.region.proj_crs)

        # Unprojected CRS should be the same
        self.assertEqual(result.buildings.data.crs, self.region.buildings.data.crs)
        self.assertEqual(self.region.buildings.data.crs, result.segments.crs)
        self.assertEqual(result.segments.crs, self.region.segments.crs)


# class TestVoronoi(unittest.TestCase):
#     def setUp(self) -> None:
#         warnings.simplefilter(
#             "ignore",
#             category=DeprecationWarning,
#         )  # HACK geopandas warning suppression

#     def test_voronoi(self):
#         from matplotlib import pyplot as plt  # noqa: I001
#         import numpy as np

#         spath = Path("./tests/test_files/test_files_segments.geojson")
#         bpath = Path("./tests/test_files/test_files_osm_buildings.geojson")

#         city = Region.load_from_files(
#             segments=spath,
#             buildings=bpath,
#             proj_crs="EPSG:3347",
#         )
#         # city.buildings.data = city.buildings.data.sample(frac=0.4, random_state=42)

#         city = city.add_pseudo_plots(building_rep="geometry", shrink=False)

#         # Assuming gdf is your GeoDataFrame with geometry columns 'geom1', 'geom2', 'geom3'
#         fig, ax = plt.subplots(figsize=(10, 10))

#         # Plot segments
#         colors1 = [tuple(np.random.rand(3)) for _ in range(len(city.segments))]
#         city.segments.plot(ax=ax, color=colors1, alpha=0.2)

#         # Plot pseudo_plots
#         pplots = gpd.GeoDataFrame(
#             geometry=city.buildings.data.pseudo_plot,
#             crs=city.proj_crs,
#         )
#         pplots = pplots.to_crs(city.segments.crs)
#         colors2 = [tuple(np.random.rand(3)) for _ in range(len(pplots))]
#         pplots.plot(ax=ax, color=colors2)

#         # Plot buildings
#         city.buildings.data.geometry.plot(
#             ax=ax,
#             color="grey",
#         )

#         # Add legend and title
#         plt.show()
