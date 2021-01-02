from dataset import TerrainDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timeit import default_timer as dt
import numpy as np
from alive_progress import alive_bar
import wandb
import os

from utils.inpainting_utils import *
from models.skip import skip


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


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "full"

    TDS_T = TerrainDataset(
        "data/MDRS/data/*.tif", dataset_type="train", randomize=False
    )
    DL_T = DataLoader(dataset=TDS_T, batch_size=1, num_workers=0)

    TDS_V = TerrainDataset(
        "data/MDRS/data/*.tif", dataset_type="validation", randomize=False
    )
    DL_V = DataLoader(dataset=TDS_V, batch_size=1, num_workers=0)

    net = skip(
        1,
        1,
        num_channels_down=[16, 32, 64, 128, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128, 128],
        num_channels_skip=[0, 0, 0, 0, 0, 0],
        filter_size_up=3,
        filter_size_down=5,
        filter_skip_size=1,
        upsample_mode="nearest",  # downsample_mode='avg',
        need1x1_up=False,
        need_sigmoid=True,
        need_bias=True,
        pad="reflection",
        act_fun="LeakyReLU",
    ).to(device)
    print(net)

    LR = 0.01
    net.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    wandb.init(project="trc-1")
    config = wandb.config
    config.learning_rate = LR
    wandb.watch(net)

    # * Information
    print(
        "# of parameters = {:,}".format(
            sum(np.prod(list(p.size())) for p in net.parameters())
        )
    )

    # * Training Loop parameters
    iterations = 1
    epochs = 300
    train = False
    reset = False

    if os.path.exists("tds-1/{}.pt".format(model_name)):
        if reset:
            os.remove("tds-1/{}.pt".format(model_name))
        else:
            checkpoint = torch.load("tds-1/{}.pt".format(model_name))
            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            loss = checkpoint["loss"]

    if train:
        try:
            with alive_bar(epochs) as bar:
                for epoch in range(epochs):
                    for i, ((x, o), y) in enumerate(DL_T):
                        data = x.to(device)
                        target = y.to(device)
                        output = net(data)
                        loss = criterion(output, target)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if publish:
                            wandb.log({"loss": loss})

                        if i % 10 == 0:
                            torch.save(
                                {
                                    "model_state_dict": net.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "loss": loss,
                                },
                                "tds-1/{}.pt".format(model_name),
                            )

                        if i + 1 == iterations:
                            break
                    bar.text("{:.15f}".format(loss.item()))
                    bar()
        except Exception as why:
            print(why)

        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            "tds-1/{}.pt".format(model_name),
        )

    if not train:
        net.eval()
        with torch.no_grad():
            for i, ((x, o), y) in enumerate(DL):
                output = net(x.to(device))
                vs = x[0].squeeze(0).cpu().numpy()
                gt = y[0].squeeze(0).cpu().numpy()
                pr = output.data[0].squeeze(0).cpu().numpy()
                draw((vs, o), gt, pr)
                plt.show()

                if i == 0:
                    break