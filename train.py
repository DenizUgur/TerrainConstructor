from dataset import TerrainDataset
import matplotlib.pyplot as plt
from model import *
from ssim import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import wandb
import sys


def revert_range(x, fix_nan=False):
    if fix_nan:
        x[x == -10] = np.nan
    x *= TDS.data_range
    x += TDS.data_min
    return x


def draw(v, gt, pr):
    vs, (ox, oy, _) = v
    vs = revert_range(vs, fix_nan=True)
    gt = revert_range(gt)
    pr = revert_range(pr)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    map_vs = ax1.contourf(vs, levels=100)
    map_gt = ax2.contourf(gt, levels=100)
    map_pr = ax3.contourf(pr, levels=100)

    ax1.scatter(ox, oy, c="red", s=25)
    ax2.scatter(ox, oy, c="red", s=25)
    ax3.scatter(ox, oy, c="red", s=25)

    fig.colorbar(map_vs, ax=ax1)
    fig.colorbar(map_gt, ax=ax2)
    fig.colorbar(map_pr, ax=ax3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    publish = False
    if len(sys.argv) > 1:
        publish = sys.argv[1] == "publish"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    TDS = TerrainDataset("data/MDRS/data/*.tif", fast_load=not publish)
    DL = DataLoader(dataset=TDS, batch_size=4, num_workers=0)
    model = FingerNet().to(device)

    if publish:
        wandb.init(project="trc-1")
        config = wandb.config
        config.learning_rate = 0.01
        wandb.watch(model)

    print(model)
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i, ((x, o), y) in enumerate(DL):
        data = x.to(device)
        target = y.to(device)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if publish:
            wandb.log({"loss": loss})
        else:
            print(
                "Iteration #{} :: Loss = {:.4f} SSIM = {:.4f}".format(
                    i + 1, loss.item(), ssim(output, target, window_size=127)
                )
            )

        if i > 250:
            vs = x[0].squeeze(0).cpu().numpy()
            gt = y[0].squeeze(0).cpu().numpy()
            pr = output.data[0].squeeze(0).cpu().numpy()
            draw((vs, o), gt, pr)
            break

    exit(0)
    model.eval()
    with torch.no_grad():
        for i, ((x, o), y) in enumerate(DL):
            output = model(x.to(device))
            if i > -1:
                vs = x[0].squeeze(0).numpy()
                gt = y[0].squeeze(0).numpy()
                pr = output.data[0].squeeze(0).cpu().numpy()
                draw((vs, o), gt, pr)
                break