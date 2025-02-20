import unittest
import warnings

import geopandas as gpd
import shapely

from urbanity.spatialgraph import SpatialGraph


class TestSpatialGraphInitIntersectsPredicate(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

    def test_create_from_three_overlapping_polygons(self):
        # Create three overlapping squares
        poly1 = shapely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = shapely.Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        poly3 = shapely.Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])

        gs = gpd.GeoSeries([poly1, poly2, poly3])
        graph = SpatialGraph.create_from_geoseries(gs)

        # Each polygon should intersect with the other two
        self.assertEqual(graph.adj_list[0], {1, 2})
        self.assertEqual(graph.adj_list[1], {0, 2})
        self.assertEqual(graph.adj_list[2], {0, 1})

    def test_create_from_touching(self):
        # Create three overlapping squares
        poly1 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])  # edge with 1
        poly2 = shapely.Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
        poly3 = shapely.Polygon([(2, 1), (3, 1), (3, 2), (2, 2)])  # corner with 1

        gs = gpd.GeoSeries([poly1, poly2, poly3])
        graph = SpatialGraph.create_from_geoseries(gs)

        # Each polygon should intersect with the other two
        self.assertEqual(graph.adj_list[0], {1})
        self.assertEqual(graph.adj_list[1], {0, 2})
        self.assertEqual(graph.adj_list[2], {1})

    def test_create_from_transitive_overlapping_polygons(self):
        # Create three squares where only adjacent ones overlap
        poly1 = shapely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = shapely.Polygon([(2, 0), (4, 0), (4, 2), (2, 2)])
        poly3 = shapely.Polygon([(4, 0), (6, 0), (6, 2), (4, 2)])

        gs = gpd.GeoSeries([poly1, poly2, poly3])
        graph = SpatialGraph.create_from_geoseries(gs)

        # First polygon only intersects second
        self.assertEqual(graph.adj_list[0], {1})
        # Middle polygon intersects both
        self.assertEqual(graph.adj_list[1], {0, 2})
        # Last polygon only intersects middle
        self.assertEqual(graph.adj_list[2], {1})

    def test_create_from_empty_geoseries(self):
        gs = gpd.GeoSeries([])
        graph = SpatialGraph.create_from_geoseries(gs)
        self.assertEqual(graph.adj_list, {})

    def test_create_from_non_sequential_index(self):
        poly1 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = shapely.Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])
        poly3 = shapely.Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])

        # Create GeoSeries with non-sequential indices
        self.geometries = {0: poly1, 2: poly2, 3: poly3}
        self.gs = gpd.GeoSeries(self.geometries)
        # Create SpatialGraph
        graph = SpatialGraph.create_from_geoseries(self.gs)

        # Check that the graph contains all original indices
        self.assertEqual(set(graph.adj_list.keys()), {0, 2, 3})


class TestSpatialGraphInitOverlapsPredicate(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

    def test_create_from_three_overlapping_polygons(self):
        # Create three overlapping squares
        poly1 = shapely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])  # overlaps 1
        poly2 = shapely.Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])  # overlaps both
        poly3 = shapely.Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])  # overlaps 1

        gs = gpd.GeoSeries([poly1, poly2, poly3])
        graph = SpatialGraph.create_from_geoseries(gs, predicate="overlaps")

        # Each polygon should intersect with the other two
        self.assertEqual(graph.adj_list[0], {1})
        self.assertEqual(graph.adj_list[1], {0, 2})
        self.assertEqual(graph.adj_list[2], {1})

    def test_create_from_touching(self):
        poly1 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])  # edge with 1
        poly2 = shapely.Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
        poly3 = shapely.Polygon([(2, 1), (3, 1), (3, 2), (2, 2)])  # corner with 1

        gs = gpd.GeoSeries([poly1, poly2, poly3])
        graph = SpatialGraph.create_from_geoseries(gs, predicate="overlaps")

        # No polygons should overlap
        self.assertEqual(graph.adj_list[0], set())
        self.assertEqual(graph.adj_list[1], set())
        self.assertEqual(graph.adj_list[2], set())

    def test_create_from_transitive_overlapping_polygons(self):
        # Create three squares where only adjacent ones overlap
        poly1 = shapely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = shapely.Polygon([(2, 0), (4, 0), (4, 2), (2, 2)])
        poly3 = shapely.Polygon([(4, 0), (6, 0), (6, 2), (4, 2)])

        gs = gpd.GeoSeries([poly1, poly2, poly3])
        graph = SpatialGraph.create_from_geoseries(gs)

        # First polygon only intersects second
        self.assertEqual(graph.adj_list[0], {1})
        # Middle polygon intersects both
        self.assertEqual(graph.adj_list[1], {0, 2})
        # Last polygon only intersects middle
        self.assertEqual(graph.adj_list[2], {1})

    def test_create_from_empty_geoseries(self):
        gs = gpd.GeoSeries([])
        graph = SpatialGraph.create_from_geoseries(gs)
        self.assertEqual(graph.adj_list, {})


class TestConnectedComponents(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppression

        adj_list = {
            0: {1, 2},
            1: {0, 2},
            2: {0, 1},
            3: set(),
        }

        self.pg = SpatialGraph(adj_list)

    def test_dfs_connected_nodes(self):
        connected = self.pg.get_connected(0)
        self.assertEqual(set(connected), {0, 1, 2})

    def test_isolated_root_node(self):
        connected = self.pg.get_connected(3)
        self.assertEqual(set(connected), {3})

    def test_invalid_root_node(self):
        with self.assertRaises(KeyError):
            self.pg.get_connected(99)


class TestCreateCCMap(unittest.TestCase):
    def test_multiple_components_mapping(self):
        # Graph with 2 components: [0,1,2] and [3,4]
        adj_list = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4], 4: [3]}
        pg = SpatialGraph(adj_list)
        expected = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
        self.assertEqual(pg.create_connected_components_map(), expected)

    def test_empty_graph_mapping(self):
        pg = SpatialGraph(adj_list={})
        self.assertEqual(pg.create_connected_components_map(), {})

    def test_isolated_nodes_mapping(self):
        # Graph with 3 isolated nodes
        adj_list = {0: [], 1: [], 2: []}
        pg = SpatialGraph(adj_list)
        expected = {0: 0, 1: 1, 2: 2}
        self.assertEqual(pg.create_connected_components_map(), expected)
