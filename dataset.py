import rasterio
import torch
from scipy.ndimage import zoom
from skimage.draw import rectangle_perimeter, line
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from glob import glob
import random


class TerrainDataset(Dataset):
    def __init__(
        self,
        dataset_glob,
        patch_size=30,
        sample_size=256,
        observer_pad=50,
        block_variance=4,
        randomize=True,
        random_state=42,
        transform=None,
    ):
        """
        dataset_glob -> glob to *.tif files (i.e. "data/MDRS/data/*.tif")
        patch_size -> the 1m^2 area to read from .TIF
        sample_size -> the 0.1m^2 res area to be trained sample size
        observer_pad -> n pixels to pad before getting a random observer
        block_variance -> how many different observer points
        randomize -> predictable randomize
        random_state -> a value that gets added to seed
        transform -> if there is any, PyTorch Transforms
        """
        np.seterr(divide="ignore", invalid="ignore")

        # * Set Dataset attributes
        self.observer_height = 0.75
        self.patch_size = patch_size
        self.sample_size = sample_size
        self.block_variance = block_variance
        self.observer_pad = observer_pad

        # * PyTorch Related Variables
        self.transform = transform

        # * Gather files
        self.files = glob(dataset_glob)
        self.randomize = randomize
        self.random_state = random_state
        if self.randomize:
            random.shuffle(self.files)

        # * Build dataset dictionary
        self.sample_dict = dict()
        start = 0
        for file in tqdm(self.files, ncols=100):
            blocks, mask = self.get_blocks(file)
            self.sample_dict[file] = {
                "start": start,
                "end": start + len(blocks[mask]),
                "mask": mask,
                "min": np.min(blocks[mask]),
                "max": np.max(blocks[mask]),
            }
            start += len(blocks[mask])

        self.sample_dict["meta"] = {
            "min": min(self.sample_dict.values(), key=lambda x: x["min"])["min"],
            "max": max(self.sample_dict.values(), key=lambda x: x["max"])["max"],
        }

        # * Dataset state
        self.current_file = None
        self.current_blocks = None

    def __len__(self):
        key = list(self.sample_dict.keys())[-2]
        return self.sample_dict[key]["end"]

    def __getitem__(self, idx):
        """
        returns (x, (ox, oy, oz)), y
        """
        rel_idx = None
        for file, info in self.sample_dict.items():
            if idx >= info["start"] and idx < info["end"]:
                rel_idx = idx - info["start"]
                if self.current_file != file:
                    b, m = self.get_blocks(file)
                    self.current_blocks = b[m]
                    self.current_file = file
                break

        meta_min = self.sample_dict["meta"]["min"]
        meta_max = self.sample_dict["meta"]["max"]
        current = self.current_blocks[rel_idx]
        current = (current - meta_min) / (meta_max - meta_min)

        adjusted = self.get_adjusted(current)
        viewshed, observer = self.viewshed(adjusted, idx)

        dataTensor = torch.from_numpy(viewshed)
        dataTensor = dataTensor.unsqueeze(0)

        targetTensor = torch.from_numpy(adjusted)
        targetTensor = targetTensor.unsqueeze(0)

        return (dataTensor, observer), targetTensor

    def viewshed(self, dem, seed):
        h, w = dem.shape
        np.random.seed(seed + self.random_state)
        rands = np.random.rand(h - self.observer_pad, w - self.observer_pad)
        template = np.zeros_like(dem)
        template[
            self.observer_pad - self.observer_pad // 2 : h - self.observer_pad // 2,
            self.observer_pad - self.observer_pad // 2 : w - self.observer_pad // 2,
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

        if self.randomize:
            np.random.seed(int(str(abs(hash(file)))[:5]) + self.random_state)
            np.random.shuffle(blocks)

        blocks = np.repeat(blocks, self.block_variance, axis=0)
        mask = ~np.isnan(blocks).any(axis=1).any(axis=1)

        return blocks, mask
