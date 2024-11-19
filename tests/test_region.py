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
from urbanity import Region

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
            segments_path=Path("./tests/test_files/test_files_segments.geojson"),
            proj_crs=self.proj_crs,
            road_network_path=Path(
                "./tests/test_files/test_files_road_network.geojson",
            ),
            buildings_path=Path("./tests/test_files/test_files_osm_buildings.geojson"),
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
        pass
