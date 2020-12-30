from dataset import TerrainDataset
import matplotlib.pyplot as plt
from model import ConvNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import wandb
import sys


def draw(v, gt, pr):
    vs, (ox, oy, _) = v
    vs[vs == -10] = np.nan
    vs *= TDS.data_range
    vs += TDS.data_min

    gt *= TDS.data_range
    gt += TDS.data_min

    pr *= TDS.data_range
    pr += TDS.data_min

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

    TDS = TerrainDataset("data/MDRS/data/*.tif", fast_load=True)
    DL = DataLoader(dataset=TDS, batch_size=1, num_workers=0)
    model = ConvNet().to(device)

    if publish:
        wandb.init(project="trc-1")
        config = wandb.config
        config.learning_rate = 0.001
        wandb.watch(model)

    print(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for i, ((x, o), y) in enumerate(DL):
        data = x.to(device)
        target = y.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if publish:
            wandb.log({"loss": loss})
        else:
            print(i, loss)

        if i > 10:
            vs = x[0].squeeze(0).numpy()
            gt = y[0].squeeze(0).numpy()
            pr = output.data[0].squeeze(0).cpu().numpy()
            draw((vs, o), gt, pr)
            break

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