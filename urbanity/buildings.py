"""Analyze buildings within a given region."""

import pathlib
import time
import warnings
from functools import reduce
from typing import Self

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

import utils

from .polyronoi import voronoiDiagram4plg
from .spatialgraph import SpatialGraph


class Buildings:
    """Container for buildings within a given region, and some analysis tools.

    Mostly a wrapper around GeoDataframes which also keeps track of projected CRS.

    Recommended construction using `Buildings.create()` intially to ensure data is fully cleaned.

    Attributes:
        data (gpd.GeoDataFrame): A geodataframe containing building *at least* building footprint data
        proj_crs (str): The crs to use when working in projected systems. Must be a value acceptable by https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input
    """

    def __init__(self, data: gpd.GeoDataFrame, proj_crs: str):
        """Initialize Buildings object.

        Adds an area column if not defined.

        Args:
            data (gpd.GeoDataFrame): A geodataframe containing building *at least* building footprint data
            proj_crs (str): The crs to use when working in projected systems. Must be a value acceptable by https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input
        """
        if data.crs != "EPSG:4326":
            data = data.to_crs("EPSG:4326")  # Default crs
        if "area" not in data:
            data["area"] = data.to_crs(proj_crs)["geometry"].area
        if "id" not in data:
            data.insert(loc=0, column="id", value=range(len(data)))

        self.data = data
        self.proj_crs = proj_crs

    @classmethod
    def create(
        cls,
        bd: gpd.GeoDataFrame | pathlib.PurePath,
        proj_crs: str,
    ) -> Self:
        """Create a buildings object from a geodataframe of footprints.

        Does the following pre-processing:
            - Explodes any multi-part geometries
            - Prunes any invalid or empty polygons
            - Dissolves any strictly overlapping polygons into one polygon. Keeping only the data from the FIRST row.

        Args:
            bd (gpd.GeoDataFrame): A geodataframe of building footprints.
            proj_crs (str): The crs to use when working in projected systems. Must be a value acceptable by https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input

        Returns:
            Self: An instance of Buildings
        """
        if not isinstance(bd, gpd.GeoDataFrame):  # TODO test coverage
            try:
                utils.load_geodf(bd)
            except:  # noqa: E722
                msg = "Must be created from a geodataframe, or a path to a parquet file that can be loaded into such."
                raise ValueError(msg) from None

        bd = bd.copy()
        bd = bd.explode(index_parts=False).reset_index()
        bd = bd[bd.is_valid]
        bd = bd[~bd.is_empty]

        bd = bd.to_crs(proj_crs)
        bd = _dissolve_overlaps(bd)
        bd = bd.to_crs("EPSG:4326")

        return Buildings(data=bd, proj_crs=proj_crs)

    @classmethod
    def load(cls, load_folder: pathlib.PurePath, proj_crs: str) -> Self:
        """Constructs a Buildings object from a previous save.

        Args:
            load_folder (pathlib.PurePath): Path to the building save folder
            proj_crs (str, optional): Main crs to use
        """
        # Load files
        gdfs = []
        for f in list(load_folder.iterdir()):
            gdf = utils.load_geodf(f)
            if f.stem != "buildings":
                gdf = gdf.rename(columns={"geometry": f.stem})
            gdfs.append(gdf)

        # Merge dataframes
        data = reduce(
            lambda left, right: pd.merge(left, right, on="id", how="inner"),  # noqa: PD015
            gdfs,
        )

        data = gpd.GeoDataFrame(data, geometry="geometry", crs=gdfs[0].crs)

        return Buildings(data=data, proj_crs=proj_crs)

    def save(self, save_folder: pathlib.PurePath) -> None:
        """Saves building data.

        Args:
            save_folder (pathlib.PurePath): Save folder location
        """
        save_folder.mkdir(parents=True)
        data = self.data
        utils.save_geodf(data, save_folder / "buildings")

    def __eq__(self, other: Self) -> bool:
        bl = self.data.equals(other.data)
        bl = bl and self.proj_crs == other.proj_crs

        return bl

    def copy(self, deep: bool = True) -> Self:
        """Returns a deepcopy by default to be consistent with pandas behaviour."""
        data = self.data.copy() if deep else self.data
        proj_crs = self.proj_crs

        return Buildings(data=data, proj_crs=proj_crs)

    def calc_floors(
        self,
        floor_height: float = 2.75,
        floor_breakpoints: list[float] | None = None,
        type_col: str | None = None,
    ) -> Self:
        """Filters buildings by volume and assigns floor counts based on height.

        Make sure units match the units of height you have in buildings.data

        Args:
            floor_height (float): Height per floor. Defaults to 2.75.
            floor_breakpoints (list[float], optional): Custom height breakpoints for
                floors. Defaults to None (will construct from floor height).
            type_col (str): The name for the building type. Defaults to "sfh"
                (single family home)

        Returns:
            Self: A copy of the Buildings with an updated `buildings` attribute, containing:
                - 'floors': Estimated floor count based on height and breakpoints or
                            floor height.

        Raises:
            AttributeError: If the `buildings` attribute is not set.
        """
        data = self.data

        # Determining Floors
        breakpoints = [-1.1, 0]  # HACK -1.1 because some data has -1 in the dataset
        max_height = data["height"].max()

        if floor_breakpoints is not None:
            breakpoints = breakpoints + floor_breakpoints

        breakpoints = breakpoints + list(
            np.arange(
                breakpoints[-1] + floor_height,
                max_height + floor_height,
                floor_height,
            ),
        )

        breakpoints = np.array(breakpoints)

        if type_col is not None:
            data["floors"] = data.apply(
                lambda r: np.nanargmax(
                    np.where(breakpoints < r.height, breakpoints, np.nan),
                )
                if r[type_col]
                else 0,
                axis=1,
            )
        else:
            data["floors"] = data["height"].map(
                lambda h: np.nanargmax(np.where(breakpoints < h, breakpoints, np.nan)),
            )

        data["floors"] = data["floors"].astype(int)

        return Buildings(data, self.proj_crs)

    def create_size_flag(
        self,
        min_vol: float,
        max_vol: float,
        flag_name: str,
        min_area: float = 0,
        max_area: float = np.inf,
    ) -> Self:
        """Selects buildings within a given volume range.

        Min volume and Max volume should use the same units as your projected CRS

        Args:
            min_vol (float): Minimum volume for filtering in cubic units
            max_vol (float): Maximum volume for filtering in cubic units
            flag_name (str): The name for the building type (e.g. "sfh")
            min_area: (float): The minimum footprint area for the building type
            max_area: (float): The maximmum footprint area for the building type

        Returns:
            object: A copy of the object with an updated `buildings` attribute, containing:
                - 'volume': Volume per building.
                - '[flag_name]': Boolean indicating if volume is within range.

        Raises:
            AttributeError: If there is no height information in the building data
        """
        data = self.data

        if "height" not in data:
            msg = 'building data does not contain a "height" column'
            raise AttributeError(msg)

        if "volume" not in data:
            data["volume"] = data["area"] * data["height"]

        # Filtering by size
        data[flag_name] = data.apply(
            lambda r: (r.volume >= min_vol)
            and (r.volume <= max_vol)
            and (r.area >= min_area)
            and (r.area <= max_area),
            axis=1,
        )

        return Buildings(data, self.proj_crs)

    def sjoin_building_features(
        self,
        other: gpd.GeoDataFrame,
        cols: list[str],
    ) -> Self:
        """Spatial join features from the most overlapping building in another dataset.

        Performs a spatial join between the building features in the current object
        and another GeoDataFrame. Retains the feature with the greatest intersection
        for each building.

        Args:
            other (gpd.GeoDataFrame): The GeoDataFrame to spatially join with the buildings.
            cols (list): A list of the column names from other that you want to add

        Returns:
            Self: A new `Buildings` object containing the joined data.
        """
        data = self.data.copy()
        other = other.copy()
        original_geom = data[["id", "geometry"]]

        data = data.to_crs(self.proj_crs)
        other = other.to_crs(self.proj_crs)

        new_data = utils.sjoin_greatest_intersection(data, other, cols)

        new_data = new_data.drop("geometry", axis=1)
        new_data = original_geom.merge(new_data, on="id", how="inner")
        new_data.crs = original_geom.crs

        return Buildings(new_data, self.proj_crs)

    def sjoin_addresses(
        self,
        ad_df: gpd.GeoDataFrame,
        join_nearest: bool = False,
        max_distance: float = 5.0,
    ) -> Self:
        """.

        Args:
            ad_df (gpd.GeoDataFrame): Adress dataframe containing *at least* point geometries for addresses
            join_nearest (bool, optional): If true, attempts to assign non-intersecting addresses to the
                nearest building within `max_distance`. Defaults to False.
            max_distance (float, optional): Used with `join_nearest = True`. Defaults to 5.0 in whatever
                 CRS is in self.proj_crs

        Returns:
            Self: _description_
        """
        data = self.data.copy()
        original_geom = data[["id", "geometry"]]

        ad_df = ad_df.copy()

        if "id" in ad_df:
            msg = "Address data cannot contain a column named 'id'"
            raise ValueError(msg)

        # Join is done in projected coordinates
        if ad_df.crs != self.proj_crs:
            warnings.warn(
                f"Address crs did not match building projected crs:{self.proj_crs}. Attempting to convert",
                stacklevel=2,
            )
            ad_df = ad_df.to_crs(self.proj_crs)
        data = data.to_crs(self.proj_crs)

        # Perform join
        new_data = gpd.sjoin(data, ad_df, how="left", predicate="contains")

        # Anecdotal evidence suggests that some address points might be *just*
        # outside the building polygons. This code assigns the points remaining
        # after the previous join to the nearest polygon
        if join_nearest:
            remaining = ad_df[~ad_df.index.isin(new_data["index_right"])]

            round1 = new_data[new_data["index_right"].notna()]
            round2 = gpd.sjoin_nearest(
                data,
                remaining,
                how="inner",
                max_distance=max_distance,
                distance_col="dist",
            )

            # Keep only the nearest matches
            round2 = round2.sort_values(["index_right", "dist"])
            round2 = round2.drop_duplicates(subset="index_right", keep="first")

            has_addr = round1["id"].to_list() + round2["id"].to_list()
            no_addr = new_data[~new_data["id"].isin(has_addr)]

            # Copy column order and reset index so concat works
            round1 = round1[new_data.columns].reset_index(drop=True)
            round2 = round2[new_data.columns].reset_index(drop=True)
            no_addr = no_addr[new_data.columns].reset_index(drop=True)
            new_data = gpd.GeoDataFrame(pd.concat([round1, round2, no_addr]))

        # Combine duplicate rows, since some buildings can have more than one
        # associated addres, all info from ad_df is combined into a list for a
        # given building id
        agg_funcs = {}
        for col in new_data:
            if col == "id":
                continue
            elif col in data:  # noqa: RET507
                agg_funcs[col] = "first"
            else:
                agg_funcs[col] = lambda x: list(set(x))

        new_data = new_data.groupby("id").agg(agg_funcs).reset_index()

        # Convert instances of [nan] to empty list
        new_cols = [col for col in new_data.columns if col not in data]
        for col in new_cols:
            new_data[col] = new_data[col].apply(
                lambda x: []
                if isinstance(x, list) and len(x) == 1 and pd.isna(x[0])
                else x,
            )

        # Output formatting
        # TODO address indices should be a list of ints, not floats
        new_data = new_data.rename(columns={"index_right": "address_indices"})
        new_data = new_data.drop("geometry", axis=1)
        new_data = original_geom.merge(new_data, on="id", how="inner")
        new_data.crs = original_geom.crs

        return Buildings(new_data, self.proj_crs)

    def create_voronoi_polygons(
        self,
        boundary: shapely.Polygon = None,
        debuff: float | None = 0.5,
        flag_col: str | None = None,
    ) -> list[tuple[2]]:
        """Generates voronoi polygons around buildings within a certain boundary.

        Make sure the boundary is the same crs as buildings.proj_crs!

        Args:
            boundary (shapely Polygon, optional): The boundary (In EPSG:4326) within which the voronoi polygons will be generated.
                Defaults to None.
            debuff (float, optional): Shrinks buildings by debuff amount. Recommended when buildings are touching.
                Defaults to 0.5.
            flag_col (str, optional): If set, buildings without the flag will be excluded when generating voronoi polygons.
                Defaults to None.
            shrink (bool, optional): If True, shrinks the boundary to approximate a convex hull around the contained buildings.
                Defaults to False.

        Returns:
            gpd.GeoSeries: A geoseries containing [building id, voronoi geometry] columns
        """
        original_crs = self.data.crs
        proj_crs = self.proj_crs

        # Get all buildings within the boundary
        if boundary is None:
            boundary = self.data["geometry"].unary_union.convex_hull

        buildings = self.data[
            self.data["geometry"].intersects(boundary, align=False)
        ].copy()
        buildings = buildings.to_crs(proj_crs)
        boundary = gpd.GeoSeries([boundary], crs="EPSG:4326")
        boundary = boundary.to_crs(proj_crs).iloc[0]

        # Filter by flagged column
        if flag_col:
            buildings = buildings[buildings[flag_col]]

        if len(buildings) == 0:
            return gpd.GeoSeries([pd.NA])

        # From this point on we only care about geometries

        # Shrink buildings a little unless otherwise specified.
        # Avoids voronoi problems with overlapping/touching buildings.
        if debuff:
            buildings["geometry"] = _shrink_buildings(
                buildings["geometry"],
                debuff_size=debuff,
                fix_multi=True,
            )

        # Voronoi
        try:
            vd = voronoiDiagram4plg(buildings, boundary, densify=True)
        except Exception as e:
            error_file = f"error_{time.strftime('%Y%m%d_%H%M%S')}.gpkg"
            msg = (
                "Voronoi generation failed. Likely due to invalid geometries."
                "Try creating your buildings object with Buildigs.create_from_dataframe() to ensure"
                "clean geometries"
            )
            buildings.to_file(error_file, driver="GPKG")
            raise RuntimeError(msg) from e

        vd = vd.to_crs(original_crs)
        vd = vd["geometry"]

        return vd


def _dissolve_overlaps(bd: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve strictly overlapping (i.e. not just touching) polygons into one polygon.

    Note that when dissolving, only the data from the *FIRST* in each group is conserved.

    Args:
        bd (gpd.GeoSeries): A geodataframe of building polygons.

    Returns:
        gpd.GeoSeries: A new GeoDataframe with overlapping polygons merged.
    """
    bd = bd.copy()

    # Generate G(V,E) graph of intersecting polygons
    # debuff slightly so touching != intersecting
    # can't use overlap because identical polys don't overlap
    pg = SpatialGraph.create_from_geoseries(
        bd["geometry"].buffer(-1e-6),
        predicate="intersects",
    )
    cc_map = pg.create_connected_components_map()

    bd["group"] = bd.index.map(cc_map)

    dissolved = bd.dissolve(by="group", aggfunc="first", as_index=False)

    dissolved["geometry"].buffer(0).reset_index()

    return dissolved.drop(columns=["group"])


def _shrink_buildings(
    geoms: gpd.GeoSeries,
    debuff_size: float,
    fix_multi: bool = False,
) -> gpd.GeoSeries:
    """Shrink buildings by debuff_size.

    Args:
        geoms (gpd.Geoseries): Building geometries
        debuff_size(float): Amount to shrink the buildings by
        fix_multi (bool, optional): If true, attempts to fix any multipolygons generated after
            shrinking by reversing the shrinking step for only those geometries. Note that this won't fix
            any multipolygons that were already present in geoms. Defaults to False.

    Returns:
        geoms (gpd.Geoseries): Adjusted geometries
    """
    original = geoms.copy()
    geoms = geoms.buffer(-debuff_size, join_style="mitre")

    if fix_multi:
        mask = geoms.apply(lambda x: isinstance(x, shapely.MultiPolygon))
        geoms[mask] = original[mask]

    return geoms
