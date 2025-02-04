import pathlib
from functools import reduce
from typing import Self

import geopandas as gpd
import pandas as pd

import utils


class Plots:
    def __init__(self, data: gpd.GeoDataFrame, proj_crs: str):
        if data.crs != "EPSG:4326":
            data = data.to_crs("EPSG:4326")  # Default crs
        if "area" not in data:
            data["area"] = data.to_crs(proj_crs)["geometry"].area
        if "id" not in data:
            data.insert(loc=0, column="id", value=range(len(data)))

        self.data = data
        self.proj_crs = proj_crs

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
        utils.save_geodf(data, save_folder / "plots")

    @classmethod
    def load(cls, load_folder: pathlib.PurePath, proj_crs: str) -> Self:
        """Constructs a Plots object from a previous save.

        Args:
            load_folder (pathlib.PurePath): Path to the building save folder
            proj_crs (str, optional): Main crs to use
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

    def __eq__(self, other: Self) -> bool:
        bl = self.data.equals(other.data)
        bl = bl and self.proj_crs == other.proj_crs

        return bl

    def copy(self, deep=True) -> Self:
        """Returns a deepcopy by default to be consistent with pandas behaviour."""
        data = self.data.copy() if deep else self.data
        proj_crs = self.proj_crs

        return Plots(data=data, proj_crs=proj_crs)
