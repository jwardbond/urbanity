"""Finds the approximate voronoi diagram generated around a polygon.

Forked from https://github.com/longavailable/voronoi-diagram-for-polygons and last
updated on 2024/12/09. Updated from source 2024/12/09.
"""

import itertools
import logging
import math
import os
import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union, voronoi_diagram


def setup_logger():
    log_file = f"worker_{os.getpid()}.log"  # Each worker gets a separate log file
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [PID %(process)d] %(levelname)s: %(message)s",
    )


def valid_comparisons(pdict: dict) -> list[list[tuple]]:
    """Given a dict, whose values are lists of polygon vertices, return cartesian product of every vertex from each polygon with every *other* polygon.

    Used to avoid cartesian products containing point combinations from within the same polygon

    Args:
        pdict (dict): a dict whose keys are polygon id numbers, and whose values are lists of tuples of polygon vertices
            e.g. [(x0, y0), (x1, y1)...].

    Returns:
        [(p1, p2), (p1, p3)...]: the cartesian product of every point from a polygon with every other point from other polygons.
    """
    result = []
    for key, tuples in pdict.items():
        other_tuples = [t for k, v in pdict.items() if k != key for t in v]
        pairs = list(itertools.product(tuples, other_tuples))
        result += pairs
    return result


def minimum_distance(gdf: gpd.GeoDataFrame) -> float | None:
    """Calculate the minimum distance between vertices of DIFFERENT geometries.

    Args:
        gdf (geopandas.GeoDataFrame): Polygons to be used.

    Returns:
        float: The minimum distance.
    """
    if len(gdf) == 1:
        return None

    # Convert to dict of poly: [(x1,y1)...] tuples
    polys = gdf.geometry.tolist()
    polys = {i: v.exterior.coords.xy for i, v in enumerate(polys)}
    polys = {k: list(zip(v[0], v[1], strict=True)) for k, v in polys.items()}

    # Generate valid comparisons
    comparisons = valid_comparisons(polys)

    # Calculate distances
    distances = [
        math.dist(p0, p1) for (p0, p1) in comparisons
    ]  # calculate distance for each pair of vertices
    nonzero_distance = [d for d in distances if d > 0.0]  # drop zeros
    return min(nonzero_distance)


def _pnts_on_line_(a: np.ndarray, spacing: float = 1.0, is_percent: bool = False):
    """Add points, at a fixed spacing, to an array representing a line.

    Sourced from https://stackoverflow.com/a/65008592/12371819.

    Args:
        a (numpy.ndarray): A sequence of points, x,y pairs, representing the bounds of a polygon or polyline object.
        spacing (float, optional): Spacing between the points to be added to the line. Defaults to 1.
        is_percent (bool, optional): Express the densification as a percent of the total length. Defaults to False.

    Returns:
        numpy.ndarray: Densified array of points.
    """
    n = len(a) - 1  # segments
    dxdy = a[1:, :] - a[:-1, :]  # coordinate differences
    leng = np.sqrt(np.einsum("ij,ij->i", dxdy, dxdy))  # segment lengths
    if is_percent:  # as percentage
        spacing = abs(spacing)
        spacing = min(spacing / 100, 1.0)
        steps = (sum(leng) * spacing) / leng  # step distance
    else:
        steps = leng / spacing  # step distance
    deltas = dxdy / (steps.reshape(-1, 1))  # coordinate steps
    pnts = np.empty((n,), dtype="O")  # construct an `O` array
    for i in range(n):  # cycle through the segments and make
        num = np.arange(steps[i])  # the new points
        pnts[i] = np.array((num, num)).T * deltas[i] + a[i]
    a0 = a[-1].reshape(1, -1)  # add the final point and concatenate
    return np.concatenate((*pnts, a0), axis=0)


def densify_polygon(gdf: gpd.GeoDataFrame, spacing="auto") -> gpd.GeoDataFrame:  # noqa: ANN001
    """Densify the vertex along the edge of polygon(s).

    Args:
        gdf (geopandas.GeoDataFrame): Polygons to be used.
        spacing (str, int, float, optional): Type or distance to be used. Defaults to 'auto'.

    Returns:
        geopandas.GeoDataFrame: A set of new polygon(s) with more vertices.

    Raises:
        ValueError: If spacing is not a string, int, or float.
    """
    if not isinstance(spacing, str | float | int):
        msg = "Spacing must be a string, int, or float."
        raise TypeError(msg)
    if isinstance(spacing, str) and (len(gdf) <= 1):
        msg = "For dataframes with length <= 1, spacing cannot be set to auto."
        raise ValueError(msg)

    if isinstance(spacing, str) and (spacing.upper() == "AUTO"):
        spacing = 0.25 * minimum_distance(gdf)  # less than 0.5? The less, the better?

    try:
        # Create a geoseries containing lists of exterior points
        ext_list = gdf["geometry"].map(lambda g: list(g.exterior.coords))

        gdf["geometry"] = ext_list.map(
            lambda x: Polygon(
                _pnts_on_line_(np.array(x).reshape(-1, 2), spacing=spacing)
            ),
        )
    except Exception as e:
        msg = "Densification failed. This is usually do to invalid geometries being encountered (empty polygons, overlapping polygons, etc.)"
        msg = f"{msg}\nOriginal error: {e!s}"
        raise type(e)(msg) from e

    return gdf


def simplify_polygon(
    geom: Polygon | MultiPolygon,
) -> Polygon | None:
    """Converts geometry to exterior polygon without holes.

    Will also transform multipolygons into polygons via exterior points

    """
    if isinstance(geom, MultiPolygon):
        # Convert each polygon in the multipolygon
        polys = [Polygon(p.exterior).buffer(0.01) for p in geom.geoms]
        polys = unary_union(polys).buffer(-0.01)
        return polys
    elif isinstance(geom, Polygon):
        return Polygon(geom.exterior)
    return None


def vertex_count_in_limit(smp: MultiPolygon, max_vertices: int) -> bool:
    polygons = list(smp.geoms) if isinstance(smp, MultiPolygon) else [smp]
    vertex_count = sum(len(poly.exterior.coords) for poly in polygons)
    logging.info(f"Processing {vertex_count} vertices")
    return vertex_count <= max_vertices


def input_warnings(gdf):
    invalids = ~gdf["geometry"].is_valid
    if sum(invalids) > 0:
        warnings.warn(
            "Invalid geometries encountered. Voronoi generation may fail.",
            stacklevel=2,
        )

    non_polys = ~(gdf["geometry"].geom_type == "Polygon")
    if sum(non_polys) > 0:
        others = list(gdf[non_polys]["geometry"].unique())
        warnings.warn(
            f"Non polygon geometries {others} encounted. Voronoi generation may fail."
        )

    empty = gdf["geometry"].is_empty
    if sum(empty) > 0:
        msg = "Empty polygons detected in input."
        raise ValueError(msg)


def voronoiDiagram4plg(  # noqa: N802
    gdf: gpd.GeoDataFrame,
    mask,  # noqa: ANN001
    debuff: float | None = None,
    densify: bool = False,
    spacing="auto",  # noqa: ANN001
) -> gpd.GeoDataFrame:
    """Create Voronoi diagram / Thiessen polygons based on polygons.

    Works on a copy of the input GeoDataFrame.

    Args:
        gdf (geopandas.GeoDataFrame): Polygons to be used to create Voronoi diagram.
        mask (GeoDataFrame, GeoSeries, (Multi)Polygon): Polygon vector used to clip the created Voronoi diagram.
        densify (bool, optional): Whether to densify the polygons. Defaults to False.
        spacing (str, int, float, optional): Spacing for densification. Defaults to 'auto'.

    Returns:
        geopandas.GeoDataFrame: Thiessen polygons.
    """
    gdf = gdf.copy()
    setup_logger()

    # input validation
    input_warnings(gdf)

    # densify
    if densify & (len(gdf) > 1):
        gdf = densify_polygon(gdf, spacing=spacing)
    gdf.reset_index(drop=True)

    # convert to MultiPolygon
    smp = gdf.unary_union

    # test for overly large inputs
    if not vertex_count_in_limit(smp, 300000):
        warnings.warn(
            "More than 300 000 vertices detected after densification. Voronoi generation not performed",
            stacklevel=2,
        )
        gdf.geometry = [pd.NA] * len(gdf)
        logging.info("\tCanceled")
        return gdf

    # create primary voronoi diagram by invoking shapely.ops.voronoi_diagram (new in Shapely 1.8.dev0)
    start = time.time()
    smp_vd = voronoi_diagram(smp)
    logging.info(f"\tVoronoi done {time.time() - start}")

    # convert to GeoSeries and explode to single polygons
    # note that it is NOT supported to GeoDataFrame directly
    gs = gpd.GeoSeries([smp_vd]).explode(index_parts=True)

    # Fix any invalid polygons
    gs.loc[~gs.is_valid] = gs.loc[~gs.is_valid].apply(lambda geom: geom.buffer(0))

    # convert to GeoDataFrame
    # note that if gdf was MultiPolygon, it has no attribute 'crs'
    gdf_vd_primary = gpd.geodataframe.GeoDataFrame(
        geometry=gs,
        crs=gdf.crs,
    ).reset_index(drop=True)

    # spatial join by intersecting and dissolve by polygon id
    gdf_vd_primary["saved_geom"] = gdf_vd_primary["geometry"]

    gdf_temp = gpd.sjoin(gdf, gdf_vd_primary, how="inner", predicate="intersects")
    gdf_temp["geometry"] = gdf_temp["saved_geom"]
    gdf_temp = gdf_temp.drop(columns=["saved_geom", "index_right"])

    start = time.time()
    gdf_temp = gdf_temp.dissolve(level=0).reset_index()
    logging.info(f"\tDissolve done {time.time() - start}")

    gdf_vd = gpd.clip(gdf_temp, mask)
    gdf_vd["geometry"] = gdf_vd["geometry"].map(simplify_polygon)

    return gdf_vd
