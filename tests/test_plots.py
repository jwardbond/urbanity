import copy
import os
import shutil
import sys
import unittest
import warnings
from pathlib import Path

import geopandas as gpd
import shapely
from geopandas.testing import assert_geodataframe_equal

from urbanity import Plots

os.environ["GDAL_DATA"] = os.path.join(
    f"{os.sep}".join(sys.executable.split(os.sep)[:-1]),
    "Library",
    "share",
    "gdal",
)  # HACK GDAL warning suppression


class TestPlots(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter(
            "ignore",
            category=DeprecationWarning,
        )  # HACK geopandas warning suppress

        # Create sample geometries
        # Simple rectangular plots
        geometries = [
            shapely.box(0, 0, 10, 10),  # 100 sq units
            shapely.box(20, 0, 25, 10),  # 50 sq units
            shapely.box(30, 0, 35, 5),  # 25 sq units
            shapely.box(40, 0, 49, 10),  # 90 sq units
        ]

        plot_data = {"id": [1, 3, 17, 21], "geometry": geometries}

        # Create GeoDataFrame with the test geometries
        self.data = gpd.GeoDataFrame(
            {"geometry": geometries, "id": range(1, len(geometries) + 1)},
            crs="EPSG:4326",  # WGS 84
        )

        # Define projection CRS for testing
        self.proj_crs = "EPSG:3347"
        self.data = gpd.GeoDataFrame(plot_data, crs=self.proj_crs)
        self.plots = Plots(data=self.data, proj_crs=self.proj_crs)
        self.save_folder = Path(__file__).parent / "test_files" / "test_plots_save"

    def test_init(self) -> None:
        plots = Plots(data=self.data, proj_crs=self.proj_crs)

        self.assertTrue("area" in plots.data)
        self.assertTrue("id" in plots.data)

        # Area calcs should be correct
        self.assertAlmostEqual(90, plots.data.iloc[3]["area"], places=5)

    def test_save_and_load(self) -> None:
        plots = copy.deepcopy(self.plots)

        if self.save_folder.exists():
            shutil.rmtree(self.save_folder)

        plots.save(self.save_folder)
        loaded = Plots.load(self.save_folder, proj_crs="EPSG:3347")

        # Files should be created
        self.assertTrue(Path(self.save_folder / "plots.parquet").exists())

        # Data should be unchanged
        assert_geodataframe_equal(loaded.data, plots.data, check_like=True)
