import rasterio
from scipy.ndimage import zoom
from tqdm import tqdm
import numpy as np


class Dataset:
    def __init__(self) -> None:
        self.sample = "data/MDRS/data/USGS_one_meter_x51y425_UT_Southern_QL1_2018.tif"

    def blockshaped(self, arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
        return (
            arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols)
        )

    def get_interpolated(self, blocks, factor=10):
        for block in tqdm(blocks, total=len(blocks), ncols=100):
            yield zoom(block, factor, order=1)

    def get_blocks(self, size):
        raster = rasterio.open(self.sample)
        grid = raster.read(1)

        # Remove minimum
        grid[grid == np.min(grid)] = np.nan

        # Find the edges to cut from
        NL = np.count_nonzero(np.isnan(grid[:, 0]))
        NR = np.count_nonzero(np.isnan(grid[:, -1]))
        NT = np.count_nonzero(np.isnan(grid[0, :]))
        NB = np.count_nonzero(np.isnan(grid[-1, :]))

        w, h = grid.shape
        if NL > NR:
            grid = grid[w % size : w, 0:h]
        else:
            grid = grid[0 : w - (w % size), 0:h]

        w, h = grid.shape
        if NT > NB:
            grid = grid[0:w, h % size : h]
        else:
            grid = grid[0:w, 0 : h - (h % size)]

        blocks = self.blockshaped(grid, size, size)
        usable_blocks = list(
            filter(lambda x: np.count_nonzero(np.isnan(x)) == 0, blocks)
        )
        usable_px = len(usable_blocks) * size * size
        all_px = np.count_nonzero(~np.isnan(grid))
        ratio = 1 - (usable_px / all_px)

        return usable_blocks, ratio