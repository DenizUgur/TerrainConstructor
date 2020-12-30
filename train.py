from dataset import TerrainDataset
import matplotlib.pyplot as plt
from model import ConvNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    TDS = TerrainDataset("data/MDRS/data/*.tif", fast_load=True)
    DL = DataLoader(dataset=TDS, batch_size=1, num_workers=0)

    # wandb.init(project="trc-1")
    # config = wandb.config
    # config.learning_rate = 0.001

    model = ConvNet().to(device)
    # wandb.watch(model)

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

        print(loss)
        # wandb.log({"loss": loss})

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        ax1.contourf(x[0].squeeze(0).numpy(), levels=100)
        ax2.contourf(y[0].squeeze(0).numpy(), levels=100)
        ax3.contourf(output.data[0].squeeze(0).cpu().numpy(), levels=100)
        plt.show()

        if i > -1:
            break

    # model.eval()
    # with torch.no_grad():
    #     for i, ((x, (ox, oy, _)), y) in enumerate(DL):
    #         output = model(x.to(device))
    #         output = output[0].squeeze(0).cpu().numpy()

    #         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    #         ax1.contourf(x[0].squeeze(0).numpy(), levels=100)
    #         ax2.contourf(y[0].squeeze(0).numpy(), levels=100)
    #         ax3.contourf(output, levels=100)

    #         ax1.scatter(ox, oy, c="red", s=25)
    #         ax2.scatter(ox, oy, c="red", s=25)
    #         ax3.scatter(ox, oy, c="red", s=25)
    #         plt.show()

    #         if i > -1:
    #             break