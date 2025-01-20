import pathlib
import warnings
from functools import reduce
from typing import Self

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

import utils

from .polyronoi import voronoiDiagram4plg


class Buildings:
    def __init__(self, data: gpd.GeoDataFrame, proj_crs: str):
        if data.crs != "EPSG:4326":
            data = data.to_crs("EPSG:4326")  # Default crs
        if "area" not in data:
            data["area"] = data.to_crs(proj_crs)["geometry"].area
        if "id" not in data:
            data.insert(loc=0, column="id", value=range(len(data)))

        self.data = data
        self.proj_crs = proj_crs

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

    def sjoin_building_features(
        self,
        df2: gpd.GeoDataFrame,
        variables: list[str],
    ) -> Self:
        """Spatial join features from the most overlapping building in another dataset.

        Performs a spatial join between the building features in the current object
        and another GeoDataFrame. Retains the feature with the greatest intersection
        for each building.

        Args:
            df2 (gpd.GeoDataFrame): The GeoDataFrame to spatially join with the buildings.
            variables (list): A list of the variables (column names) from df2 that you want to add

        Returns:
            Self: A new `Buildings` object containing the joined data.
        """
        data = self.data.copy()
        df2 = df2.copy()
        original_geom = data[["id", "geometry"]]

        data = data.to_crs(self.proj_crs)
        df2 = df2.to_crs(self.proj_crs)

        new_data = utils.sjoin_greatest_intersection(data, df2, variables)

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
        data = self.data.copy()
        original_geom = data[["id", "geometry"]]

        ad_df = ad_df.copy()

        data = data.to_crs(self.proj_crs)

        # Address df should be in projected crs

        if ad_df.crs != self.proj_crs:
            warnings.warn(
                f"Address crs did not match building projected crs:{self.proj_crs}. Attempting to convert",
                stacklevel=2,
            )
            ad_df.to_crs(self.proj_crs)

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
        new_data = new_data.rename(columns={"index_right": "address_indices"})
        new_data = new_data.drop("geometry", axis=1)
        new_data = original_geom.merge(new_data, on="id", how="inner")
        new_data.crs = original_geom.crs

        return Buildings(new_data, self.proj_crs)

    def create_voronoi_plots(
        self,
        boundary: shapely.Polygon = None,
        flag_col: str | None = None,
        shrink: bool = True,
        building_rep: str = "mrr",
    ) -> list[tuple[2]]:
        """Make sure the boundary is the same crs as buildings.proj_crs!.

        Args:
            boundary (shapely Polygon, optional): The boundary within which the voronoi polygons will be generated.
                Defaults to None (Uses the convex hull of all buildings).
            flag_col (str, optional): If set, buildings without the flag will be excluded when generating voronoi polygons.
                Defaults to None.
            shrink (bool, optional): If True, shrinks the boundary to approximate a convex hull around the contained buildings.
                Defaults to False.
            building_rep (str, optional): The representation to use for the buildings. Options are "mrr" (minimum rotated rectangle)
                "geometry" (default geometry)

        Returns:
            A list of (geometry, building_id) tuples representing the voronoi polygons for each building.
        """
        buildings = self.data[["id", "geometry"]]
        original_crs = buildings.crs
        buildings = buildings.to_crs(self.proj_crs)

        # If no boundary, just make the boundary the convex hull of all buildings
        if boundary is None:
            boundary = buildings["geometry"].unary_union.convex_hull

        # Get all buildings within the boundary
        buildings = buildings[
            buildings["geometry"].intersects(boundary, align=False)
        ]  # HACK

        if len(buildings) == 0:
            return (np.nan, np.nan)

        if flag_col:
            buildings = buildings[buildings[flag_col]].copy()

        # Simplify buildings
        if building_rep == "mrr":
            buildings["geometry"] = buildings["geometry"].minimum_rotated_rectangle()
        elif building_rep == "geometry":
            pass  # TODO
        else:
            msg = f"building_rep={building_rep} is not supported"
            raise AttributeError(msg)

        # Shrink boundary to the concave hull of the two most streetfacing (closest to the boundary)
        if shrink:
            boundary = shapely.concave_hull(
                buildings["geometry"].unary_union,
                ratio=0.8,
            )

        # Voronoi
        vd = voronoiDiagram4plg(buildings, boundary, densify=True)
        vd = vd.to_crs(original_crs)
        vd = vd[["id", "geometry"]]
        vd = list(vd.itertuples(index=False, name=None))

        return vd

    #
    # UTILTIY METHODS
    #
    def save(self, save_folder: pathlib.PurePath) -> None:
        """Saves building data.

        Args:
            save_folder (pathlib.PurePath): Save folder location
        """
        save_folder.mkdir(parents=True)
        data = self.data
        utils.save_geodf(data, save_folder / "buildings")

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

    def __eq__(self, other: Self) -> bool:
        bl = self.data.equals(other.data)
        bl = bl and self.proj_crs == other.proj_crs

        return bl

    def copy(self, deep=True) -> Self:
        """Returns a deepcopy by default to be consistent with pandas behaviour."""
        data = self.data.copy() if deep else self.data
        proj_crs = self.proj_crs

        return Buildings(data=data, proj_crs=proj_crs)
