"""Classes and methods for dealing with "plots": polygonal boundaries demarcating property ownership."""

import math
import pathlib
import warnings
from functools import reduce
from typing import Self

import geopandas as gpd
import pandas as pd
from shapely.constructive import maximum_inscribed_circle

import utils


class Plots:
    """A collection of plots, and methods for operating on the same.

    Attributes:
        data: Plot data. Stored as a geodataframe in EPSG:4326.
        proj_crs: The CRS used for tasks which require projection.
    """

    # ****************
    # CONSTRUCTION
    # ****************
    def __init__(self, data: gpd.GeoDataFrame, proj_crs: str):
        """Initializes a Plots object.

        Args:
            data (gpd.GeoDataFrame): Plot data.
            proj_crs (str): CRS to use for tasks that require projection.
        """
        if data.crs != "EPSG:4326":
            data = data.to_crs("EPSG:4326")  # Default crs
        if "area" not in data:
            data["area"] = data["geometry"].to_crs(proj_crs).area
        if "id" not in data:
            data.insert(loc=0, column="id", value=range(len(data)))

        self.data = data
        self.proj_crs = proj_crs

    @classmethod
    def load(cls, load_folder: pathlib.PurePath, proj_crs: str) -> Self:
        """Constructs a Plots object from a previous save.

        Args:
            load_folder (pathlib.PurePath): Path to the building save folder
            proj_crs (str, optional): Main crs to use

        Returns:
            Plots (Plots): A newly constructed Plots object.
        """
        # Load files
        gdfs = []
        for f in list(load_folder.iterdir()):
            gdf = utils.load_geodf(f)
            if f.stem != "plots":
                gdf = gdf.rename(columns={"geometry": f.stem})
            gdfs.append(gdf)

        # Merge dataframes
        data = reduce(
            lambda left, right: pd.merge(left, right, on="id", how="inner"),  # noqa: PD015
            gdfs,
        )

        data = gpd.GeoDataFrame(data, geometry="geometry", crs=gdfs[0].crs)

        return Plots(data=data, proj_crs=proj_crs)

    @classmethod
    def create(cls, data: gpd.GeoDataFrame | pathlib.PurePath, proj_crs: str) -> Self:
        """Create a Plots object from a geodataframe of land plots.

        TODO decide what cleaning methods are needed
        Args:
            data (gpd.GeoDataFrame): A geodataframe of land plots. Will get converted to EPSG:4326
            proj_crs (str): The crs to use when working with projected geometries. Must be a value acceptable by https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input

        Returns:
            Self: An instance of Plots
        """
        if not isinstance(data, gpd.GeoDataFrame):  # TODO test coverage
            try:
                utils.load_geodf(data)
            except:  # noqa: E722
                msg = "Must be created from a geodataframe, or a path to a parquet file that can be loaded into such."
                raise ValueError(msg) from None

        data = data.copy()

        return Plots(data=data, proj_crs=proj_crs)

    # ****************
    # METHODS
    # ****************
    def update(
        self,
        other: Self,
        cols: list[str] | None,
        overwrite: bool = False,
    ) -> Self:
        """Updates the columns in Plots.data with columns from other.data by joining on the "id" columns.

        Columns guaranteed to be conserved:
            - `id`
            - `geometry`
            - `area` (because geometry is conserved)

        Other columns are conserved unless they are overwritten by columns in `other`.

        Useful (for example) when you want to update a Plots object with the results of some geometric operations, but you don't want to update the actual geometries.

        Args:
            other (Plots): A Plots object

        Returns:
            Self: A copy of the original Plots object updated with the specified data from Other
        """
        return self.update_from_gdf(other.data, cols, overwrite)

    def update_from_gdf(
        self,
        other: gpd.GeoDataFrame,
        cols: list[str] | None,
        overwrite: bool = False,
    ) -> Self:
        """Updates the columns in Plots.data with data from a dataframe by joining on the "id" columns.

        Columns guaranteed to be conserved:
            - `id`
            - `geometry`
            - `area` (because geometry is conserved)

        Other columns are conserved unless they are overwritten by columns in `other`.

        Useful (for example) when you want to update a Plots object with the results of some geometric operations, but you don't want to update
        the actual geometries.

        Args:
            other (gpd.GeoDataFrame): A geodataframe containing (at least) a column labelled "id"

        Returns:
            Self: A copy of the original Plots object updated with the specified data from other.
        """
        self_data = self.data.copy()
        other_data = other.copy()

        if "id" not in other_data:
            msg = "'other' has no id column"
            raise KeyError(msg)

        if cols is None:
            cols = other_data.columns.to_list()

        cols = [c for c in cols if c not in ["id", "area", "geometry"]]

        overwritten = [c for c in cols if c in self_data.columns]

        if len(overwritten) > 0:
            if not overwrite:
                msg = "Update would overwrite a column. Please correctly specify cols or set overwrite=True."
                raise ValueError(msg)
            self_data = self_data.drop(columns=overwritten)

        merge_cols = ["id", *cols]
        other_data = other_data[merge_cols]

        self_data = self_data.merge(other_data, on="id", how="left")

        return Plots(self_data, self.proj_crs)

    def subtract_polygons(
        self,
        polygons: gpd.GeoDataFrame,
    ) -> Self:
        """Subtract polygons from plots.

        This is basically a wrapper for geopandas overlay diff.

        Args:
            polygons (gpd.GeoDataframe): Geodataframe of polygons to subtract

        Returns:
            Region: Returns a new Region object
        """
        # Parse inputs
        data = self.data.copy()

        if polygons.crs != "EPSG:4326":
            warnings.warn(
                "Input data not in EPSG:4326. Attempting to convert.",
                stacklevel=2,
            )
            polygons = polygons.to_crs("EPSG:4326")

        # Get set difference
        data = data.overlay(polygons, how="difference")

        # Need to recalculate area
        data["area"] = data.to_crs(self.proj_crs).area

        return Plots(data=data, proj_crs=self.proj_crs)

    def create_circle_fit_flag(
        self,
        radius: float,
        tolerance: float | None = None,
        flag_name: str = "circle_fit",
    ) -> Self:
        """Determines whether or not a plot can fit a circle of a given radius.

        Uses Shapely's maximum inscribed circle method. See here: https://shapely.readthedocs.io/en/latest/reference/shapely.maximum_inscribed_circle.html

        Args:
            radius (float): Plots that can fit maximum inscribed circles above this radius will be flagged as True.
                Must be in units consistent with Plots.proj_crs.
            tolerance (float, optional): Tolerance/precision to use when finding MIC center.
                Defaults to None (max(width, height) / 1000)
            flag_name (str, optional): Name of the resulting flag column in Plots.data. Defaults to "circle_fit".

        Returns:
            Plots: A newly created Plots instance, with plots.data containing the new flag.
        """
        # Prune everything with an area < circle area into separate DF, flag = False
        if radius <= 0:
            msg = "Radius must be strictly greater than 0"
            raise ValueError(msg)

        data = self.data.copy()
        ori_crs = data.crs

        circle_area = math.pi * radius**2

        # Exclude anything that obviously wont fit.
        too_small = data["area"] < circle_area
        pre_processed = data[too_small].copy()
        remaining = data[~too_small].copy()

        pre_processed[flag_name] = False

        # Use vectorized shapely MIC methods to flag remaining geoms
        mic_gs = maximum_inscribed_circle(
            remaining["geometry"].to_crs(self.proj_crs),
            tolerance,
        )
        mic_rads = mic_gs.length
        mic_rads = mic_rads.fillna(-1.0)
        remaining[flag_name] = mic_rads > radius

        # Reform data
        data = pd.concat([pre_processed, remaining], ignore_index=True)
        data = gpd.GeoDataFrame(data, geometry="geometry", crs=ori_crs)

        data = data.set_index("id", drop=False)
        data.index.name = None

        return Plots(data=data, proj_crs=self.proj_crs)

    def create_predicate_flag(
        self,
        polygons: gpd.GeoDataFrame,
        predicate: str,
        flag_name: str = "pred_flag",
    ) -> Self:
        """Flag all plots that satisfy the given predicate.

        Args:
            polygons (gpd.GeoDataFrame): Data to test (plots.data.contains(input))
            predicate (str): The binary predicate to use
            flag_name (gpd.GeoDataFrame, optional): The name of the new column. Defaults to "contains"

        Returns:
            Plots: A newly created Plots object, with a
        """
        polygons = polygons.copy()
        data = self.data.copy()

        if flag_name in data:
            msg = "Column pred_flag already in plots. Specify a unique flag_name"
            raise ValueError(msg)

        if polygons.crs != data.crs:
            msg = f"polygons crs {polygons.crs} doesn't match Plots crs {data.crs}, attempting to convert"
            warnings.warn(msg, stacklevel=2)

            polygons = polygons.to_crs(data.crs)

        joined = data.sjoin(polygons, how="inner", predicate=predicate)
        data[flag_name] = data["id"].isin(joined["id"])

        return Plots(data=data, proj_crs=self.proj_crs)

    def sjoin_most_intersecting(
        self,
        buildings: gpd.GeoDataFrame,
        cols: list[str],
    ) -> Self:
        """Spatial join features from the most overlapping building in another dataset.

        Performs a spatial join between the building features in the current object
        and another GeoDataFrame. Retains the feature with the greatest intersection
        for each building.

        This is useful when you want to do something like: flag all plots containing a
        building labelled as "single family home".

        Args:
            buildings (gpd.GeoDataFrame): The GeoDataFrame to spatially join with plots
            cols (list): A list of the column names from buildings that you want to add

        Returns:
            Self: A new `Buildings` object containing the joined data.
        """
        data = self.data.copy()
        buildings = buildings.copy()
        original_geoms = data[["id", "geometry"]]

        data = data.to_crs(self.proj_crs)
        buildings = buildings.to_crs(self.proj_crs)

        new_data = utils.sjoin_greatest_intersection(data, buildings, cols)

        new_data = new_data.drop("geometry", axis=1)
        new_data = original_geoms.merge(new_data, on="id", how="inner")
        new_data.crs = original_geoms.crs

        return Plots(new_data, self.proj_crs)

    # ****************
    # UTILTIY
    # ****************
    def save(self, save_folder: pathlib.PurePath) -> None:
        """Saves building data.

        Args:
            save_folder (pathlib.PurePath): Save folder location
        """
        save_folder.mkdir(parents=True)
        data = self.data
        utils.save_geodf(data, save_folder / "plots")

    def __eq__(self, other: Self) -> bool:
        bl = self.data.equals(other.data)
        bl = bl and self.proj_crs == other.proj_crs

        return bl

    def copy(self, deep=True) -> Self:
        """Returns a deepcopy by default to be consistent with pandas behaviour.

        Args:
            deep (bool, optional): If True, returns a deep copy of Self. Defaults to True.

        Returns:
            Plots (Plots): A newly constructed Plots object.
        """
        data = self.data.copy() if deep else self.data
        proj_crs = self.proj_crs

        return Plots(data=data, proj_crs=proj_crs)
