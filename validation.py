from dataset import TerrainDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import wandb
from tensorboardX import SummaryWriter
import os


from model import UNet

if __name__ == "__main__":
    writer = SummaryWriter()
    TDS_T = TerrainDataset(
        "data/MDRS/data/*.tif",
        dataset_type="validation",
        randomize=False,
        fast_load=True,
        limit_samples=1,
    )
    trainLoader = DataLoader(dataset=TDS_T, batch_size=1, num_workers=0)

    model = UNet(1, 1, depth=5, merge_mode="add").cuda()
    print(model)
    criterion = nn.MSELoss()

    wandb_name = "amber-haze-273"
    if os.path.exists(f"trc-1/model-{wandb_name}.pt"):
        checkpoint = torch.load(f"trc-1/model-{wandb_name}.pt")
        model.load_state_dict(checkpoint["model_state_dict"])

    (dummy, _), _ = TDS_T[0]
    writer.add_graph(model, dummy.unsqueeze(0).cuda())
    for (x, o), y in trainLoader:
        x = x.cuda()
        y = y.cuda()
        out = model(x)
        loss = criterion(out, y)
        print(loss)

        x = x[0].squeeze(0).cpu().numpy()
        x[x == TerrainDataset.NAN] = np.nan
        y = y[0].squeeze(0).cpu().numpy()
        pr = out[0].detach().squeeze(0).cpu().numpy()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(x)
        ax2.imshow(y)
        ax3.imshow(pr)
        plt.show()
    writer.close()
