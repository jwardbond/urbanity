"""Efficiently parallellize cpu-bound map functions on geoseries."""

import concurrent.futures as cf
import gc
import multiprocessing
from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm


class GeoParallel:
    """Parallelization methods for geoseries.

    Attributes:
        n_workers (int, optional): Number of processors to use. Defaults to number of CPU - 1.
        prog_bar (bool, optional): If True, prints a progress bar. Defaults to False.
    """

    def __init__(self, n_workers: int | None = None, prog_bar: bool = False):
        """Initializes GeoParallel instance.

        Args:
            n_workers (int | None, optional): Number of worker processes. Defaults to CPU count - 1
            prog_bar (bool, optional): Whether to show progress bar
        """
        self.n_workers = n_workers if n_workers else multiprocessing.cpu_count() - 1
        self.prog_bar = prog_bar

    @staticmethod
    def _mapped_func_wrapper(chunk: gpd.GeoSeries, func: callable) -> gpd.GeoSeries:
        """Apply a function to every element in a chunk."""
        return chunk.map(func)

    def apply_chunked(
        self,
        gs: gpd.GeoSeries,
        func: callable,
        n_chunks: int | None = None,
        desc: str = "Chunked apply",
    ) -> pd.Series:
        """Apply function to GeoSeries in parallel chunks.

        Args:
            gs: Input GeoSeries
            func: Function to apply to each element
            n_chunks: Number of chunks. Defaults to n_workers * 4
            desc: Progress description

        Returns:
            Processed Series/GeoSeries
        """
        if n_chunks is None:
            n_chunks = min(len(gs), self.n_workers * 4)  # 4 chunks per worker

        chunks = [gs.loc[idx] for idx in np.array_split(gs.index, max(1, n_chunks))]
        wrapped_func = partial(self._mapped_func_wrapper, func=func)

        results = self._parallelize(chunks, wrapped_func, desc)
        gc.collect()
        return pd.concat(results, ignore_index=True)

    def _parallelize(self, items: list, func: callable, desc: str) -> list:
        """Parallelizes func over items with yielding results for memory efficiency.

        Args:
            items: Array of items to apply func to
            func: Function to parallelize func(item)
            desc: Process description (for progress bar)

        Rerturns:
            a list containing iteration results
        """
        with cf.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            jobs = {executor.submit(func, item): idx for idx, item in enumerate(items)}

            iterator = cf.as_completed(jobs)

            results = [None] * len(items)
            if self.prog_bar:
                with tqdm(total=len(jobs), desc=desc) as pbar:
                    for future in cf.as_completed(jobs):
                        idx = jobs[future]
                        results[idx] = future.result()
                        del jobs[future]
                        pbar.update(1)
                        gc.collect()
            else:
                for future in iterator:
                    idx = jobs[future]
                    results[idx] = future.result()
                    del jobs[future]
                    gc.collect()

        return results
