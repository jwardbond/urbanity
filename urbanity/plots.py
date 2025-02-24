"""Classes and methods for dealing with "plots": polygonal boundaries demarcating property ownership."""

import math
import pathlib
from functools import reduce
from typing import Self

import geopandas as gpd
import pandas as pd
import shapely
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

    # ****************
    # METHODS
    # ****************
    def subtract_polygons(
        self,
        polygons: gpd.GeoDataFrame,
    ) -> Self:
        """Subtracts polygons from plots.

        This is basically a wrapper for geopandas overlay diff.

        Args:
            polygons (geopandas.GeoDataframe): Geodataframe of polygons to subtract

        Returns:
            Region: Returns a new Region object
        """
        # Parse inputs
        data = self.data.copy()

        # Get set difference
        data = data.overlay(polygons, how="difference")
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
        remaining[flag_name] = mic_rads > radius + tolerance

        # Reform data
        data = pd.concat([pre_processed, remaining], ignore_index=True)
        data = gpd.GeoDataFrame(data, geometry="geometry", crs=self.proj_crs)

        data = data.set_index("id", drop=False)
        data.index.name = None

        return Plots(data=data, proj_crs=self.proj_crs)

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

        return Plots(data=data, proj_crs=proj_crs)()
