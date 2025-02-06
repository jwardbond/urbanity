import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm


class GeoParallel:
    def __init__(self, n_jobs: int | None = None, prog_bar: bool = False):
        self.n_jobs = n_jobs if n_jobs else multiprocessing.cpu_count() - 1
        self.prog_bar = prog_bar

    def apply(
        self,
        gs: gpd.GeoSeries,
        func: callable,
        desc: str = "Parallel apply",
    ) -> gpd.GeoSeries:
        results = self._parallelize(gs, func, desc)
        return gpd.GeoSeries(results)

    @staticmethod
    def _mapped_func_wrapper(item: gpd.GeoSeries, func: callable):
        return item.map(func)

    def apply_chunked(
        self,
        gs: gpd.GeoSeries,
        func: callable,
        chunk_size: int = 1000,
        desc="Chunked apply",
    ) -> gpd.GeoSeries:
        chunks = [
            gs.loc[idx]
            for idx in np.array_split(gs.index, max(1, len(gs) // chunk_size))
        ]

        wrapped_func = partial(self._mapped_func_wrapper, func=func)

        results = self._parallelize(chunks, wrapped_func, desc)
        return gpd.GeoSeries(pd.concat(results, ignore_index=True))

    def _parallelize(self, items, func, desc) -> list:
        """Parallelizes func over items.

        Args:
            items (_type_): Array of items to apply func to
            func (_type_): _Function to parallelize func(item)
            desc (_type_): _Process description (for progress bar)

        Returns:
            list: a list of results with the same order as items
        """
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            if self.prog_bar:
                result = list(
                    tqdm(executor.map(func, items), total=len(items), desc=desc),
                )
                return result
            return list(executor.map(func, items))

            # future_to_index = {
            #     executor.submit(func, item): i for i, item in enumerate(items)
            # }

            # if self.prog_bar:
            #     with tqdm(total=total, desc=desc) as pbar:
            #         for future in as_completed(future_to_index):
            #             results.append((future_to_index[future], future.result()))
            #             pbar.update(1)
            # else:
            #     for future in as_completed(future_to_index):
            #         results.append((future_to_index[future], future.result()))

            # results = sorted(results, key=lambda x: x[0])
            # return [r[1] for r in results]
