import pathlib
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

    def create_voronoi_plots(
        self,
        boundary: shapely.Polygon = None,
        min_building_footprint: float = 0,
        shrink: bool = True,
        building_rep: str = "mrr",
    ) -> list[tuple[2]]:
        """Make sure the boundary is the same crs as buildings.proj_crs!.

        Args:
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
        buildings = buildings[buildings["geometry"].within(boundary)]

        if len(buildings) == 0:
            return (np.nan, np.nan)

        # Remove buildings smaller than the min_building_footprint
        tb = buildings.to_crs(self.proj_crs)
        tb = tb[tb["geometry"].area >= min_building_footprint][["id"]]
        buildings = buildings.merge(tb, how="right", on="id")  # filtering join

        # Simplify buildings
        if building_rep == "mrr":
            buildings["geometry"] = buildings["geometry"].minimum_rotated_rectangle()
        elif building_rep == "geometry":
            pass  # TODO
        else:
            msg = f"building_rep={building_rep} is not supported"
            raise AttributeError(msg)

        # Shrink boundary to the concave hull of the two most streetfacing (closest to the boundary)
        # points of each building #TODO remove commented code
        if shrink:
            # if building_rep != "mrr":
            #     warnings.warn(
            #         "shrink=True and building type != MRR, this may mess up plot generation.",
            #         stacklevel=2,
            #     )
            # boundary_points = []
            # for _, r in buildings.iterrows():
            #     boundary_points += utils.order_closest_vertices(
            #         r[building_rep],
            #         boundary,
            #     )[:2]

            # boundary = shapely.concave_hull(shapely.MultiPoint(boundary_points), 1)
            boundary = shapely.concave_hull(
                buildings["geometry"].unary_union,
                ratio=0.8,
            )

        # Voronoi
        vd = voronoiDiagram4plg(buildings, boundary, densify=True)
        vd = vd.to_crs(original_crs)
        vd = list(vd.itertuples(index=False, name=None))

        return vd

    #
    # UTILTIY METHODS
    #
    def save(self, save_folder: pathlib.PurePath) -> None:
        """Saves building data.

        The saves created with this method can be loaded using Buildings.load_from_save.
        Each column that contains shapely data is saved as a separate file.

        Args:
            save_folder (pathlib.PurePath): Save folder location
        """
        save_folder.mkdir(parents=True)
        data = self.data

        # Save any column with shapely data that isn't the "geometry" column
        geometry_columns = []
        for c in data.columns:
            if (
                isinstance(data[c].iloc[0], shapely.geometry.base.BaseGeometry)
                and c != "geometry"
            ):
                gdf = gpd.GeoDataFrame(
                    {"id": data["id"], "geometry": data[c]},
                    crs=data.crs,
                )
                utils.save_geodf(gdf, save_folder / c)
                geometry_columns.append(c)

        # Save all other data
        utils.save_geodf(
            data.drop(columns=geometry_columns),
            save_folder / "buildings",
        )

    @classmethod
    def load(cls, load_folder: pathlib.PurePath, proj_crs: str) -> Self:
        """Constructs a Buildings object from a previous save.

        Calls utils.load_gdf so save file type can be abstracted. Returned object will have EPSG:4326 crs by default.

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
