import os
import sys
import pathlib

import numpy as np
import geopandas as gpd


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


def load_geojson(
    x: str | pathlib.PurePath,
):
    """Util function used to parse function inputs."""
    if type(x) is str:
        x = pathlib.Path(x)

    x = gpd.read_file(x, engine="pyogrio", use_arrow=True)
    x.set_crs("EPSG:4326", allow_override=False)
    x["id"] = x["id"].astype(str).astype(np.int64)  # since id loads as an object
    x = x.fillna(value=np.nan)

    return x


def save_geodf_with_prompt(x: gpd.GeoDataFrame, savepath: pathlib.PurePath):
    if savepath.exists():  # then prompt for overwrite
        prompt_success = False
        while not prompt_success:
            overwrite = str(input(f"Overwriting {savepath}. Proceed? (Y/N)"))
            if overwrite == "Y" or overwrite == "y":
                prompt_success = True
                savepath.write_text(x.to_json())
            elif overwrite == "N" or overwrite == "n":
                prompt_success = True
                sys.exit("Exiting")
    else:
        savepath.parent.mkdir(parents=True, exist_ok=True)  # make parent folder
        savepath.write_text(
            x.to_json(to_wgs84=True)
        )  # save new file with standard coords


class HiddenPrints:
    """Used with `with` to hide print messages"""

    def __enter__(self):
        self._origina_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._origina_stdout
