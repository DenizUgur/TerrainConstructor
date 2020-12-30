from dataset import TerrainDataset
from model import ConvNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    TDS = TerrainDataset("data/MDRS/data/*.tif")
    DL = DataLoader(dataset=TDS, batch_size=1, num_workers=0)

    wandb.init(project="trc-1")
    config = wandb.config
    config.learning_rate = 0.001

    model = ConvNet().to(device)
    wandb.watch(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for i, ((x, _), y) in enumerate(DL):
        data = x.to(device)
        target = y.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss})