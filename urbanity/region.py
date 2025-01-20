import copy
from pathlib import PurePath
from typing import Self

import geopandas as gpd
import pandas as pd
import shapely
import shapely.ops
import swifter
from genregion import generate_regions

import utils

from .buildings import Buildings

# TODO add saving
# TODO add adjacency attribute
# TODO clarify docstring for returning Self / obj
# TODO Add crs checking in init
# TODO make returning more sensible, will require figuring out correct copy logic
# TODO there is some real fucky copying stuff happening here


class Region:
    """The functional unit of urbanity: a region divided into neighbourhood "segments" using the road network.

    Segments are a vector-based, polygonal representation of a geographic area generated from road networks according to the code outlined `here <https://github.com/PaddlePaddle/PaddleSpatial/blob/main/paddlespatial/tools/genregion/README.md`.

    Attributes:
        segments(gpd.GeoDataFrame): A geodataframe containing (at least) the polygon segments of a given region.
        road_network(gpd.GeoDataFrame): A geodataframe containing the road network
        proj_crs(str): The crs used for anything that requires projection, the value can be anything accepted by `pyroj <https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input>` such as an authority string (eg "EPSG:4326") or a WKT string.
    """

    def __init__(
        self,
        segments: gpd.GeoDataFrame,
        proj_crs: str,
        road_network: gpd.GeoDataFrame = None,
        buildings: Buildings = None,
    ):
        if "area" not in segments:
            segments["area"] = segments.to_crs(proj_crs)["geometry"].area

        if "id" not in segments:
            segments.insert(
                loc=0,
                column="id",
                value=range(len(segments)),
            )

        self.proj_crs = proj_crs
        self.segments = segments
        self.road_network = road_network
        self.buildings = buildings  # property w _buildings

    @property
    def buildings(self) -> Buildings:
        if self._buildings is None:
            msg = "Buildings data has not been set"
            raise AttributeError(msg)
        return self._buildings

    @buildings.setter
    def buildings(self, obj: Buildings):
        if obj is None:
            self._buildings = None
        else:
            if type(obj) is not Buildings:
                msg = "value must be a Buildings object"
                raise TypeError(msg)
            if obj.data.crs != "EPSG:4326":
                msg = "Building data must be in EPSG:4326"
                raise ValueError(msg)

            self._buildings = obj
            self._buildings.proj_crs = self.proj_crs

    #
    # CONSTRUCTORS
    #
    @classmethod
    def build_from_network(
        cls,
        network: gpd.GeoDataFrame | gpd.GeoSeries,
        proj_crs: str,
        grid_size: int = 1024,
        min_area: int = 10000,
        max_area: int = 0,
        width_thres: int = 20,
        clust_width: int = 25,
        point_precision: int = 2,
    ) -> Self:
        """Creates segments from a road network.

        Creates a Segments object from a simplified road network. This is basically a wrapper for the code outlined `here <https://github.com/PaddlePaddle/PaddleSpatial/blob/main/paddlespatial/tools/genregion/README.md`. The betwork must be in the WGS84 / EPSG:4326 crs.

        For more information, see:

            `A Scalable Open-Source System for Segmenting Urban Areas with Road Networks <https://dl.acm.org/doi/10.1145/3616855.3635703>` and

        Args:
            network (GeoDataFrame or GeoSeries or Purepath): The road network to use, stored as a geodataframe/geoseries of linestrings OR the path to a .geojson containing the same.
            proj_crs (str): The crs used for anything that requires projection, the value can be anything accepted by `pyroj <https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input>` such as an authority string (eg "EPSG:4326") or a WKT string.
            grid_size (int, optional):Passed to _create_segments for segmentation. Used to build a grid dictionary for searching. Defaults to 1024.
            min_area (int, optional): Passed to _create_segments for segmentation. The minimum area of a generated region. Defaults to 10000.
            max_area (int, optional): Passed to _subdivide_segments for segmentation. The maximum area of a generated region. Defaults to 0 (no max area).
            width_thres (int, optional): Passed to _create_segments for segmentation. The minimum ratio of area/perimeter. Defaults to 20.
            clust_width (int, optional): Passed to _create_segments for segmentation. The threshold that helps construct the cluster.
            point_precision (int, optional): Passed to _create_segments for segmentation. The precision of the point object while processing.

        Returns:
            Region: An instance of Region with the segments in WGS84/EPSG:4326
        """
        # Convert to projected crs\
        network = network.to_crs(proj_crs)
        edges = network["geometry"].to_list()

        # Extract polygons
        # print("Segmenting road network...", end=" ")
        with utils.HiddenPrints():
            urban_regions = generate_regions(
                edges,
                grid_size=grid_size,
                area_thres=min_area,
                width_thres=width_thres,
                clust_width=clust_width,
                point_precision=point_precision,
            )
        # print(utils.PrintColors.OKGREEN + "Done" + utils.PrintColors.ENDC)

        segments = gpd.GeoDataFrame(geometry=urban_regions, crs=proj_crs)

        if max_area:
            segments = cls._subdivide_segments(segments, max_area)

        # Convert back to default crs
        segments = segments.to_crs("EPSG:4326")

        segments["id"] = segments.index
        segments = segments[["id", "geometry"]]

        return Region(segments, proj_crs, road_network=network)

    @classmethod
    def load_from_files(
        cls,
        segments: PurePath,
        proj_crs: str,
        road_network: PurePath | None = None,
        buildings: PurePath | None = None,
    ) -> Self:
        """Creates a Region object using saved .geojson files for the relevant attributes.

        Args:
            segments (PurePath): The path to the .geojson containing pre-made segments
            proj_crs (str): The crs used for anything that requires projection, the value can be anything accepted by
                `pyroj <https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input>`
                such as an authority string (eg "EPSG:4326") or a WKT string.
            road_network(PurePath | None, optional): Path to the `.geojson` file for the road network.
                Defaults to None.
            buildings (PurePath | None, optional): Path to the `.geojson` file for buildings.
                Defaults to None.

        Returns:
            Region: An instance of `Region` with the segments in WGS84/EPSG:4326
        """
        segments = utils.load_geodf(segments)

        if road_network:
            road_network = utils.load_geodf(road_network)

        if buildings:
            buildings = utils.load_geodf(buildings)
            buildings = Buildings(buildings, proj_crs)

        return Region(segments, proj_crs, road_network, buildings)

    @classmethod
    def load(cls, load_folder: PurePath, proj_crs: str) -> Self:
        # Load segments
        try:
            s = next(load_folder.glob("segments.*"))
            segments = utils.load_geodf(s)
        except StopIteration as e:
            msg = f"Save not found at {load_folder!s}"
            raise FileNotFoundError(msg) from e

        # Load road networks
        r = next(load_folder.glob("road_network.*"), None)
        road_network = utils.load_geodf(r) if r else None

        # Load buildings
        if (load_folder / "buildings").exists():
            buildings = Buildings.load(load_folder / "buildings", proj_crs)
        else:
            buildings = None

        return Region(
            segments=segments,
            proj_crs=proj_crs,
            road_network=road_network,
            buildings=buildings,
        )

    @classmethod
    def _subdivide_segments(
        cls,
        segments: gpd.GeoDataFrame,
        max_area: int,
    ) -> gpd.GeoDataFrame:
        """Subdivides all segments greater than a minimum area.

        Overly large segments are divided in half either vertically or horizontally until they are below the max_area.

        Args:
            segments (geopandas.Geodataframe): A geodataframe containing the segments
            max_area (int): The maximum area of a segment. Units will depend on the value of `Region.proj_crs`

        Returns:
            segments (geopandas.Geodataframe): A geodataframe containing the subdivided segments.
        """
        # TODO Figure out how to handle buildings on edges.
        segments = copy.deepcopy(segments)

        larger = segments[segments.area > max_area].copy()
        smaller = segments[segments.area <= max_area].copy()

        while not larger.empty:
            # Split large geometries
            larger["geo_tmp"] = larger.apply(
                lambda row: cls._split_polygon(row.geometry),
                axis=1,
            )
            larger = larger.explode(column="geo_tmp", ignore_index=True)
            larger["geometry"] = larger["geo_tmp"]

            # Combine the split dataframe again
            segments = gpd.GeoDataFrame(pd.concat([larger, smaller]))
            segments["area"] = segments.geometry.area

            # Generate new dataframe splits
            larger = segments[segments.area > max_area].copy()
            smaller = segments[segments.area <= max_area].copy()

        # re-index
        segments = segments.drop(labels=["geo_tmp"], axis=1)
        segments = segments.reset_index(drop=True)

        return segments

    @classmethod
    def _split_polygon(cls, geom: shapely.Polygon) -> list[shapely.Polygon]:
        """Splits a polygon in half either vertically or horizontally.

        Args:
            geom (shapely.Polygon): A shapely polygon

        Returns:
            geoms (list): A list of shapely polygons
        """
        bounds = geom.bounds

        # If geometry is longer than it is tall, split along a vertical line
        if (bounds[2] - bounds[0]) > (bounds[3] - bounds[1]):
            x_mid = (bounds[0] + bounds[2]) / 2
            splitter = shapely.LineString([(x_mid, bounds[1]), (x_mid, bounds[3])])

        # Else, split along a horizontal line
        else:
            y_mid = (bounds[1] + bounds[3]) / 2
            splitter = shapely.LineString([(bounds[0], y_mid), (bounds[2], y_mid)])

        # Convert from geometry collection to list
        geoms = shapely.ops.split(geom, splitter)
        geoms = list(geoms.geoms)
        return geoms

    #
    # SEGMENTS
    #
    def subtract_polygons(
        self,
        polygons: gpd.GeoDataFrame,
    ) -> Self:
        """Subtracts polygons from segments.

        This is basically a wrapper for geopandas overlay diff.

        Args:
            polygons (geopandas.GeoDataframe): Geodataframe of polygons to subtract

        Returns:
            Region: Returns a new Region object
        """
        # Parse inputs
        segments = self.segments.copy()

        # Get set difference
        segments = segments.overlay(polygons, how="difference")
        segments["area"] = segments.to_crs(self.proj_crs).area

        return Region(
            segments=segments,
            proj_crs=self.proj_crs,
            road_network=self.road_network,
            buildings=self._buildings,
        )

    def flag_segments_by_buildings(
        self,
        flag_name: str,
        building_flag: str,
        threshold_pct: float,
        threshold_num: int,
    ) -> Self:
        if building_flag not in self.buildings.data:
            msg = f"No flag named {building_flag} found in building data"
            raise KeyError(msg)

        # TODO keep in mind that the following is a common operation and might be good to abstract so
        # I am not running it all the time
        segments = self.segments.copy()
        buildings = self.buildings.data[["geometry", building_flag]]

        joined = segments[["id", "geometry"]].sjoin(
            buildings,
            how="left",
            predicate="intersects",
        )
        # end todo

        # Calculate the sum and fraction of buildings within each segment
        joined = joined.groupby("id", as_index=False).agg(
            mean_flag=(f"{building_flag}", "mean"),
            sum_flag=(f"{building_flag}", "sum"),
        )

        joined[flag_name] = joined.apply(
            lambda r: (r.sum_flag >= threshold_num) and (r.mean_flag >= threshold_pct),
            axis=1,
        )

        # drop useless columns and rejoin
        joined = joined.drop(columns=["mean_flag", "sum_flag"])

        segments = segments.merge(joined, on="id", how="left")

        return Region(
            segments=segments,
            proj_crs=self.proj_crs,
            road_network=self.road_network,
            buildings=self._buildings,
        )

    def agg_features(
        self,
        polygons: gpd.GeoDataFrame,
        feature_name: str,
        how: str = "mean",
        fillnan: str | None = None,
    ) -> Self:
        """Given segments and a geodataframe of polygons, aggregate the polygons on a per-segment basis.

        For example, if you have building footprints + height data, you can calculate the average height of all buildings within each segment with `how="mean"`.

        Args:
            polygons (gpd.GeoDataFrame | PurePath): A geodataframe containing features and the polygons to aggregate over
            feature_name (str): The feature (column) name within "polygons" to aggregate
            how (str, optional): The desired aggregation behaviour. Options are "mean". Defaults to "mean"
            fillnan (_type_, optional):  Value to fill NaN entries with. Defaults to `None`

        Raises:
            ValueError: An unsupported aggregation behaviour (`how`) was specified

        Returns:
            Region: A copy of the `Region` object after aggregation
        """
        # Parse inputs
        segments = self.segments.copy()

        polygons = polygons.copy()

        # Join
        right_gdf = polygons[["geometry", feature_name]]
        joined = segments.sjoin(right_gdf, how="left").drop("index_right", axis=1)

        if how == "mean":
            joined = joined.groupby("id")[feature_name].mean()
        else:
            msg = "How must be one of: mean"
            raise ValueError(msg)

        segments = segments.merge(joined, on="id")

        if fillnan is not None:
            segments = segments.fillna(value=fillnan)

        return Region(
            segments=segments,
            proj_crs=self.proj_crs,
            road_network=self.road_network,
            buildings=self._buildings,
        )

    def disagg_features(
        self,
        gdf: gpd.GeoDataFrame,
        feature_name: str,
        how: str = "area",
    ) -> Self:
        """#TODO finish docstring.

        Args:
            gdf (gpd.GeoDataFrame): _description_
            feature_name (str): _description_
            how (str, optional): _description_. Defaults to "area".

        Raises:
            ValueError: _description_

        Returns:
            Self: _description_
        """
        # Parse inputs
        segments = self.segments.copy()
        gdf = gdf[["geometry", feature_name]]

        # Change to projected crs
        original_geom = segments[["id", "geometry"]]
        segments = segments.to_crs(self.proj_crs)
        gdf = gdf.to_crs(self.proj_crs)

        # Intersect dataframes
        if how == "area":
            gdf["area"] = gdf["geometry"].area

            # Split the gdf by segment boundaries
            # dropping the area temporarily just helps with naming
            split_gdf = gpd.overlay(
                gdf,
                segments.drop(labels=["area"], axis=1),
                how="intersection",
            )

            # Split feature proportional to area
            split_gdf["split_area"] = split_gdf["geometry"].area
            fname = f"split_{feature_name}"

            split_gdf[fname] = (
                split_gdf[feature_name] * split_gdf["split_area"] / split_gdf["area"]
            )

            # Join back to original df
            split_gdf = split_gdf[["id", fname]]
            grouped = split_gdf.groupby("id")[fname].sum()
            segments = segments.merge(grouped, on="id", how="inner")

            # change new column to original name
            segments = segments.rename(columns={fname: feature_name})

        else:
            msg = f"how = {how} is not a valid argument"
            raise ValueError(msg)

        # change back to original crs
        segments = segments.drop("geometry", axis=1)
        segments = original_geom.merge(segments, on="id", how="inner")
        segments.crs = original_geom.crs

        return Region(
            segments=segments,
            proj_crs=self.proj_crs,
            road_network=self.road_network,
            buildings=self._buildings,
        )

    def add_pseudo_plots(
        self,
        segment_flag: str = "",
        progress_bar: bool = False,
        **kwargs,
    ) -> Self:
        """TODO.

        Args:
            segment_flag (str, optional): Only generate plots for segments flagged with this flag. Defaults to "" (all segments).
            progress_bar (bool, optional): True will print a swifter progress bar. Defaults to False
            **kwargs: Other arguments that will be passed to Buildings.create_voronoi_plots.

        Returns:
            Region: a new Region object
        """
        buildings = self.buildings.copy()

        # Filter segments
        segments = (
            self.segments[self.segments[segment_flag]]
            if segment_flag
            else self.segments
        )
        segments = segments.copy()

        # Generate list of voronoi polygons for each segment
        # note that this is done in the buildings' proj_crs
        # (which should be the regions proj_crs as well)
        def vpoly_apply(boundary: shapely.Polygon) -> list[tuple]:
            polys = buildings.create_voronoi_plots(boundary=boundary, **kwargs)
            return polys

        segments["voronoi_polys"] = (
            segments["geometry"]
            .to_crs(buildings.proj_crs)
            # .swifter.progress_bar(progress_bar)
            .apply(vpoly_apply)
        )

        # Extract the voronoi polygons and associated building ids
        # by exploding and converting tuples to new columns (in a new df)
        exploded = segments.explode("voronoi_polys")

        voronoi_df = pd.DataFrame(
            exploded["voronoi_polys"].tolist(),
            columns=["id", "pseudo_plot"],
        )

        buildings.data = buildings.data.merge(voronoi_df, how="left", on="id")

        return Region(
            segments=self.segments,
            proj_crs=self.proj_crs,
            buildings=Buildings(buildings.data, self.proj_crs),
        )

    #
    # UTILTIY METHODS
    #
    def save(self, save_folder: PurePath) -> None:
        """Saves a Region object as a collection of geojson.

        If the folder already exists, then it appends a number. Use load_region constructor function to reverse.

        Args:
            save_folder(pathlib.PurePath): Path to the save folder.
        """
        # Make save folder
        i = 0
        while save_folder.exists():
            i += 1
            save_folder = save_folder.parent / f"{save_folder.stem}_{i}"
        save_folder.mkdir(parents=True)

        # Save
        utils.save_geodf(self.segments, save_folder / "segments")

        if self.road_network is not None:
            utils.save_geodf(self.road_network, save_folder / "road_network")
        if self.buildings is not None:
            self.buildings.save(save_folder / "buildings")

    def __eq__(self, other: Self) -> bool:
        bl = self.segments.equals(other.segments)
        bl = bl and self.proj_crs == other.proj_crs

        if self._buildings:
            bl = bl and other._buildings and self.buildings == other.buildings

        return bl
