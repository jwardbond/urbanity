import pathlib
from typing import Self

import numpy as np
import geopandas as gpd


class Buildings:
    def __init__(self, data: gpd.GeoDataFrame, proj_crs: str):
        if "area" not in data:
            data["area"] = data.to_crs(proj_crs)["geometry"].area
        if "id" not in data:
            data.insert(loc=0, column="id", value=range(len(data)))

        data = data.to_crs("EPSG:4326")  # default crs

        self.data = data
        self.proj_crs = proj_crs

    @classmethod
    def read_geojson(cls, data: pathlib.PurePath, proj_crs: str):
        data = gpd.read_file(data)
        data.set_crs("EPSG:4326", allow_override=False)
        data["id"] = (
            data["id"].astype(str).astype(np.int64)
        )  # since id loads as an object
        data = data.fillna(value=np.nan)

        return cls(data, proj_crs)

    def create_volume_flag(
        self, min_vol: float, max_vol: float, flag_name: str
    ) -> Self:
        """Selects buildings within a given volume range.

        Min volume and Max volume should use the same units as your projected CRS

        Args:
            min_vol (float): Minimum volume for filtering in cubic units
            max_vol (float): Maximum volume for filtering in cubic units
            flag_name (str): The name for the building type (e.g. "sfh")

        Returns:
            object: A copy of the object with an updated `buildings` attribute, containing:
                - 'volume': Volume per building.
                - '[flag_name]': Boolean indicating if volume is within range.

        Raises:
            AttributeError: If there is no height information in the building data
        """
        data = self.data

        if "height" not in data:
            raise AttributeError('building data does not contain a "height" column')

        if "volume" not in data:
            data["volume"] = data["area"] * data["height"]

        # Filtering by volume
        data[flag_name] = data["volume"].map(
            lambda x: (x >= min_vol) and (x <= max_vol)
        )

        return Buildings(data, self.proj_crs)

    def calc_floors(
        self,
        floor_height: float = 2.75,
        floor_breakpoints: list[float] = None,
        type_col: str = None,
    ):
        """Filters buildings by volume and assigns floor counts based on height.

        Make sure units match the units of height you have in buildings.data

        Args:
            floor_height (float): Height per floor. Defaults to 2.75.
            floor_breakpoints (list[float], optional): Custom height breakpoints for floors. Defaults to None (will construct from floor height).
            type_col (str): The name for the building type. Defaults to "sfh" (single family home)

        Returns:
            object: A copy of the object with an updated `buildings` attribute, containing:
                - 'floors': Estimated floor count based on height and breakpoints or floor height.

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
                breakpoints[-1] + floor_height, max_height + floor_height, floor_height
            )
        )

        breakpoints = np.array(breakpoints)

        if type_col is not None:
            data["floors"] = data.apply(
                lambda r: np.nanargmax(
                    np.where(breakpoints < r.height, breakpoints, np.nan)
                )
                if r[type_col]
                else 0,
                axis=1,
            )
        else:
            data["floors"] = data["height"].map(
                lambda h: np.nanargmax(np.where(breakpoints < h, breakpoints, np.nan))
            )

        data["floors"] = data["floors"].astype(int)

        return Buildings(data, self.proj_crs)
