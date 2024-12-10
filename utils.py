import os
import pathlib
import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


class PrintColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def load_geodf(
    x: str | pathlib.PurePath,
) -> gpd.GeoDataFrame:
    """Util function used to parse function inputs."""
    if type(x) is str:
        x = pathlib.Path(x)

    x = gpd.read_file(x, engine="pyogrio", use_arrow=True)
    x.set_crs("EPSG:4326", allow_override=False)
    x["id"] = x["id"].astype(str).astype(np.int64)  # since id loads as an object
    x = x.fillna(value=np.nan)

    return x


def save_geodf(x: gpd.GeoDataFrame, savepath: pathlib.PurePath) -> None:
    """Util function for saving geopandas files without prompts."""
    i = 0
    while savepath.exists():  # then prompt for overwrite
        i += 1
        savepath = savepath.parent / f"{savepath.stem}_{i}.{savepath.suffix}"

    savepath.parent.mkdir(parents=True, exist_ok=True)  # make parent folder
    savepath.write_text(
        x.to_json(to_wgs84=True),
    )  # save new file with standard coords


def save_geodf_with_prompt(x: gpd.GeoDataFrame, savepath: pathlib.PurePath) -> None:
    """Util function for saving geopandas files."""
    if savepath.exists():  # then prompt for overwrite
        prompt_success = False
        while not prompt_success:
            overwrite = str(input(f"Overwriting {savepath}. Proceed? (Y/N)"))
            if overwrite in ("Y", "y"):
                prompt_success = True
                savepath.write_text(x.to_json())
            elif overwrite in ("N", "n"):
                prompt_success = True
                sys.exit("Exiting")
    else:
        savepath.parent.mkdir(parents=True, exist_ok=True)  # make parent folder
        savepath.write_text(
            x.to_json(to_wgs84=True),
        )  # save new file with standard coords


def sjoin_greatest_intersection(
    target_df: gpd.GeoDataFrame,
    source_df: gpd.GeoDataFrame,
    variables: list[str],
) -> gpd.GeoDataFrame:
    """Join variables from source_df based on the largest intersection. In case of a tie it picks the first one.

    From https://pysal.org/tobler/_modules/tobler/area_weighted/area_join.html#area_join

    Args:
        source_df (geopandas.GeoDataFrame): GeoDataFrame containing source values.
        target_df (geopandas.GeoDataFrame): GeoDataFrame containing target values.
        variables (str or list-like): Column(s) in `source_df` dataframe for variable(s) to be joined.

    Returns:
        geopandas.GeoDataFrame: The `target_df` GeoDataFrame with joined variables as additional columns.

    """
    if not pd.api.types.is_list_like(variables):
        variables = [variables]

    for v in variables:
        if v in target_df.columns:
            msg = f"Column '{v}' already present in target_df."
            raise ValueError(msg)

    target_df = target_df.copy()
    target_ix, source_ix = source_df.sindex.query(
        target_df.geometry,
        predicate="intersects",
    )
    areas = (
        target_df.geometry.values[target_ix]  # noqa: PD011
        .intersection(source_df.geometry.values[source_ix])  # noqa: PD011
        .area
    )

    main = []
    for i in range(len(target_df)):  # vectorise this loop?
        mask = target_ix == i
        if np.any(mask):
            main.append(source_ix[mask][np.argmax(areas[mask])])
        else:
            main.append(np.nan)

    main = np.array(main, dtype=float)
    mask = ~np.isnan(main)

    for v in variables:
        arr = np.empty(len(main), dtype=object)
        arr[mask] = source_df[v].to_numpy()[main[mask].astype(int)]
        try:
            arr = arr.astype(source_df[v].dtype)
        except TypeError:
            warnings.warn(
                f"Cannot preserve dtype of '{v}'. Falling back to `dtype=object`.",
            )
        target_df[v] = arr

    return target_df


def order_closest_vertices(
    p: shapely.Polygon,
    target: shapely.Polygon,
) -> list:
    """Order the exterior vertices of a polygon, according to their proximity to the boundaries of another polygon.

    Args:
        p (shapely.Polygon): _description_
        target (shapely.Polygon): _description_

    Returns:
        list: a list of exterior vertices of p, ordered by their proximity to target
    """
    ordered_points = []  # (point, distance)

    target = target.exterior

    for vertex in list(p.exterior.coords):
        v = shapely.Point(vertex)
        distance = target.distance(v)
        ordered_points.append((v, distance))

    # Remove duplicate points
    ordered_points = list(set(ordered_points))

    ordered_points = sorted(ordered_points, key=lambda x: x[1])

    return [x[0] for x in ordered_points]


class HiddenPrints:
    """Used with `with` to hide print messages."""

    def __enter__(self):
        self._origina_stdout = sys.stdout
        sys.stdout = pathlib.Path.open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        sys.stdout.close()
        sys.stdout = self._origina_stdout
