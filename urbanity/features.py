import math
from pathlib import Path, PurePath

import shapely
import numpy as np
import geopandas as gpd
import shapely.affinity

import utils


def subtract_polygons(
    segments: gpd.GeoDataFrame | PurePath,
    polygons: gpd.GeoDataFrame | PurePath,
    savefolder: PurePath = None,
):
    """Subtract polygons from segments

    This is basically a wrapper for geopandas overlay diff. Works on a copy of inputs.

    Args:
        segments (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of segments or path to a .geojson containing them.
        polygons (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of polygons to subtract, or path to a .geojson containing them.
        savepath (pathlib.PurePath, optional): Folder to save result to. Defaults to None (not saved)

    Returns:
        geopandas.GeoDataframe: The input segments less the polygons
    """

    # Parse inputs
    segments = segments.copy()
    polygons = polygons.copy()

    segments = utils.input_to_geodf(segments)
    polygons = utils.input_to_geodf(polygons)

    # Get set difference
    segments = segments.overlay(polygons, how="difference")

    # Save / overwrite segment file
    if savefolder:
        savepath = savefolder / (savefolder.stem + "_segments.geojson")
        utils.save_geodf_with_prompt(segments, savepath)

    return segments


def agg_features(
    segments: gpd.GeoDataFrame | PurePath,
    polygons: gpd.GeoDataFrame | PurePath,
    feature: str,
    how="mean",
    fillnan=None,
    savefolder: PurePath = None,
):
    """Given segments and a geodataframe of polygons, aggregate the polygons on a per-segment basis.

    For example, if you have building footprints + height data, you can calculate the average height of all buildings within each segment with `how="mean"`

    Args:
        segments (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of segments or path to a .geojson containing them.
        polygons (geopandas.GeoDataframe, pathlib.PurePath):
        savefolder (pathlib.PurePath, optional): Folder to save result to. Defaults to None (not saved)

    Returns:
        geopandas.GeoDataframe: A geodataframe containing the set difference segments - polygons
    """
    # TODO fill out docstring

    # Parse inputs
    segments = segments.copy()
    polygons = polygons.copy()

    segments = utils.input_to_geodf(segments)
    polygons = utils.input_to_geodf(polygons)

    # Join
    right_gdf = polygons[["geometry", feature]]
    joined = segments.sjoin(right_gdf, how="left").drop("index_right", axis=1)

    if how == "mean":
        joined = joined.groupby("id")[feature].mean()
    else:
        raise ValueError("How must be one of: mean")

    segments = segments.merge(joined, on="id")

    if fillnan is not None:
        segments = segments.fillna(value=fillnan)

    # Save / overwrite segment file
    if savefolder:
        savepath = savefolder / (savefolder.stem + "_segments.geojson")
        utils.save_geodf_with_prompt(segments, savepath)

    return segments


def add_tiles(
    segments: gpd.GeoDataFrame | PurePath,
    savefolder: PurePath = None,
):
    """Given segments and a list of polygons representing areas to NOT include, add various metrics

    Args:
        segments (geopandas.GeoDataframe, pathlib.PurePath): Geodataframe of segments or path to a .geojson containing them.
        savefolder (pathlib.PurePath, optional): Folder to save result to. Defaults to None (not saved)

    Returns:
        geopandas.GeoDataframe: A geodataframe containing the set difference segments - polygons
    """

    # Parse inputs
    segments = segments.copy()
    segments = utils.input_to_geodf(segments)

    # Project
    segments = segments.to_crs("EPSG:3857")

    # Get minimum rotated bounding rectangles
    segments["mrr"] = segments.minimum_rotated_rectangle()

    # Rotate rectangles to be axis-aligned
    segments["mrr_angle"] = segments["mrr"].apply(lambda mrr: _mrr_azimuth(mrr))
    segments["mrr"] = segments.apply(
        lambda row: shapely.affinity.rotate(
            row.mrr, -1 * row.angle, "center", use_radians=True
        ),
        axis=1,
    )

    # Add tiles
    segments[["p1", "p2", "p3", "p4"]] = segments.apply(
        lambda row: _tile_rect(row.mrr, 3, 0), axis=1, result_type="expand"
    )

    # Clip tiles to segments and select best tiling
    segments["tmp_geom"] = segments.apply(
        lambda row: shapely.affinity.rotate(
            row.geometry, -1 * row.mrr_angle, "center", use_radians=True
        ),
        axis=1,
    )

    segments[["best_tiling", "n_tiles"]] = segments.apply(
        lambda row: _filter_multipolygon(
            row.tmp_geom, [row.p1, row.p2, row.p3, row.p4]
        ),
        axis=1,
        result_type="expand",
    )
    segments = segments.drop(labels=["p1", "p2", "p3", "p4", "tmp_geom"], axis=1)

    # Rotate the best tiling to match the original geometry
    segments["best_tiling"] = segments.apply(
        lambda row: shapely.affinity.rotate(
            row.best_tiling, row.mrr_angle, "center", use_radians=True
        ),
        axis=1,
    )


def _mrr_azimuth(mrr: shapely.Polygon):
    """Calculates the azimuth of a rectangle on a cartesian plane.

    In other words. this is the angle (w.r.t. the eastern direction) that you would need to rotate and axis-aligned
    rectangle in order to get your rectangle.

    Adapted from `this SO answer <https://stackoverflow.com/questions/66108528/angle-in-minimum-rotated-rectangle>`

    Args:
        mrr (shapely.Polygon): A Shapely polygon representing a minimum rotated rectangle
    Returns:
        The azimuthal angle in radians.
    """
    bbox = list(mrr.exterior.coords)
    axis1 = np.linalg.norm(np.subtract(bbox[0], bbox[3]))
    axis2 = np.linalg.norm(np.subtract(bbox[0], bbox[1]))

    if axis1 <= axis2:
        az = _vector_azimuth(bbox[0], bbox[1])
    else:
        az = _vector_azimuth(bbox[0], bbox[3])

    return az


def _vector_azimuth(point1: tuple, point2: tuple):
    """Calculates the azimuth between two points.

    In other words, if you are standig on the plane looking east, this is the angle you have to turn
    in order to point the same direction as the vector connecting the two points.

    Adapted from `this SO answer <https://stackoverflow.com/questions/66108528/angle-in-minimum-rotated-rectangle>`

    Returns:
        The azimuthal angle in radians.
    """

    angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
    return angle if angle > 0 else angle + np.pi


def _tile_rect(mrr: shapely.Polygon, size: float, margin: float = 0):
    """Tiles an axis-aligned rectangle with squares of a given size

    Returns four different multipolygons, aligned with each corner of the minimum rotated rectangle.

    Args:
        mrr (shapely.Polygon): A Shapely polygon representing a minimum rotated rectangle
        size (Float): The size of the square to use when tiling the MRR

    Returns:
        p1, p2, p3, p4 (geopandas.GeoSeries): tiled rectangles as multipolygons.
    """
    buffered_size = size + 2 * margin

    # Tile the shape, starting with the bottom left corner
    xmin, ymin, xmax, ymax = mrr.bounds

    w = xmax - xmin
    h = ymax - ymin

    ncols, col_remainder = divmod(w, buffered_size)
    nrows, row_remainder = divmod(h, buffered_size)

    # Tile the shape, starting with the bottom left corner
    boxes = []
    for row in range(int(nrows)):
        for col in range(int(ncols)):
            bl_x = col / ncols * w + xmin + margin  # bottom left x coord of the box
            bl_y = row / nrows * h + ymin + margin
            tr_x = bl_x + size
            tr_y = bl_y + size

            box = shapely.box(
                xmin=bl_x,
                ymin=bl_y,
                xmax=tr_x,
                ymax=tr_y,
            )
            boxes.append(box)

    p1 = shapely.MultiPolygon(boxes)
    p2 = shapely.affinity.translate(p1, col_remainder)  # shift right
    p3 = shapely.affinity.translate(p1, row_remainder)  # shif up
    p4 = shapely.affinity.translate(p1, col_remainder, row_remainder)

    return p1, p2, p3, p4


# TODO fix this.... very slow
def _filter_multipolygon(bounding_poly, inner_polys):
    best_intersections = []
    for i, inner_poly in enumerate(inner_polys):
        intersections = []
        for j, p in enumerate(inner_poly.geoms):
            if bounding_poly.contains(p):
                intersections.append(p)
        if len(intersections) > len(best_intersections):
            best_intersections = intersections

    return shapely.MultiPolygon(best_intersections), len(best_intersections)


if __name__ == "__main__":
    filepath = Path("data/east_york_ontario/")
