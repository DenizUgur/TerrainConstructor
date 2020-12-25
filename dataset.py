import rasterio
from scipy.ndimage import zoom
from skimage.draw import rectangle_perimeter, line
from tqdm import tqdm
import numpy as np
from glob import glob
import random


class Dataset:
    def __init__(self, patch_size=30, sample_size=256, pad=50, block_variance=4, randomize=True):
        """
        patch_size -> the 1m^2 area to read from .TIF
        sample_size -> the 0.1m^2 res area to be trained sample size
        block_variance -> how many different observer points
        pad -> n pixels to pad before getting a random observer
        """
        np.seterr(divide="ignore", invalid="ignore")

        # * Set Dataset attributes
        self.observer_height = 0.75
        self.patch_size = patch_size
        self.sample_size = sample_size
        self.block_variance = block_variance
        self.pad = pad

        # * Gather files
        self.files = glob("data/MDRS/data/*.tif")
        self.randomize = randomize
        if self.randomize:
            random.shuffle(self.files)

        # * Keep track of requested files
        self.last_req_index = -1
        self.blocks = None

    def prepare(self):
        if self.blocks is None or self.last_req_index == len(self.blocks):
            # * Load new file
            if len(self.files) == 0:
                return False

            self.blocks = self.get_blocks(self.files.pop())
            if self.randomize:
                random.shuffle(self.blocks)

        return True

    def get(self, index=None, batch=None, return_observer=False):
        """
        returns Viewsheded and Ground Truth
        and optionally observer point
        """
        if index is None:
            self.last_req_index += 1
            index = self.last_req_index

        adjusted = self.get_adjusted(self.blocks[index])
        viewsheded, observer = self.viewshed(adjusted, index)

        if return_observer:
            return viewsheded, adjusted, observer
        return viewsheded, adjusted

    def viewshed(self, dem, seed):
        h, w = dem.shape
        np.random.seed(seed)
        rands = np.random.rand(h - self.pad, w - self.pad)
        template = np.zeros_like(dem)
        template[
            self.pad - self.pad // 2 : h - self.pad // 2,
            self.pad - self.pad // 2 : w - self.pad // 2,
        ] = rands
        observer = tuple(np.argwhere(template == np.max(template))[0])

        yp, xp = observer
        zp = dem[observer] + self.observer_height
        observer = (xp, yp, zp)
        viewshed = np.copy(dem)

        # * Find perimiter
        rr, cc = rectangle_perimeter((1, 1), end=(h - 2, w - 2), shape=dem.shape)

        # * Iterate through perimiter
        for yc, xc in zip(rr, cc):
            # * Form the line
            ray_y, ray_x = line(yp, xp, yc, xc)
            ray_z = dem[ray_y, ray_x]

            m = (ray_z - zp) / np.hypot(ray_y - yp, ray_x - xp)

            max_so_far = -np.inf
            for yi, xi, mi in zip(ray_y, ray_x, m):
                if mi < max_so_far:
                    viewshed[yi, xi] = np.nan
                else:
                    max_so_far = mi

        return viewshed, observer

    def blockshaped(self, arr, nside):
        """
        Return an array of shape (n, nside, nside) where
        n * nside * nside = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nside == 0, "{} rows is not evenly divisble by {}".format(h, nside)
        assert w % nside == 0, "{} cols is not evenly divisble by {}".format(w, nside)
        return (
            arr.reshape(h // nside, nside, -1, nside)
            .swapaxes(1, 2)
            .reshape(-1, nside, nside)
        )

    def get_adjusted(self, block):
        _, block = block
        zoomed = zoom(block, 10, order=1)
        y, x = zoomed.shape
        startx = x // 2 - (self.sample_size // 2)
        starty = y // 2 - (self.sample_size // 2)
        return zoomed[
            starty : starty + self.sample_size, startx : startx + self.sample_size
        ]

    def get_blocks(self, file):
        if self.patch_size % 2 == 0:
            self.patch_size += 1

        raster = rasterio.open(file)
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
            grid = grid[w % self.patch_size : w, 0:h]
        else:
            grid = grid[0 : w - (w % self.patch_size), 0:h]

        w, h = grid.shape
        if NT > NB:
            grid = grid[0:w, h % self.patch_size : h]
        else:
            grid = grid[0:w, 0 : h - (h % self.patch_size)]

        blocks = self.blockshaped(grid, self.patch_size)
        usable_blocks = list(
            filter(lambda x: np.count_nonzero(np.isnan(x)) == 0, blocks)
        )

        # * Increase variation
        final_blocks = []
        for block in usable_blocks:
            for _ in range(self.block_variance):
                final_blocks.append((len(final_blocks), block))

        return final_blocks
