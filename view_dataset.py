from dataset import TerrainDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
        )
        self.layer2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
        )
        self.layer3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=2),
        )
        self.layer4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


if __name__ == "__main__":
    num_epochs = 35
    batch_size = 25
    learning_rate = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    TDS = TerrainDataset("data/MDRS/data/*.tif")
    DL = DataLoader(dataset=TDS, batch_size=3, num_workers=0)

    #! CNN
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i, ((x, _), y) in enumerate(DL):
        data = x.to(device)
        target = y.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.squeeze(1))
        loss.backwards()
        optimizer.step()

        _, prediction = torch.max(output.data, 1)
        npp = prediction.squeeze(0).numpy()
        print("Done")

    for (x, (ox, oy, _)), y in DS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 18))
        ax1.contourf(x, levels=100)
        ax2.contourf(y, levels=100)

        ax1.scatter(ox, oy, c="red", s=25)
        ax2.scatter(ox, oy, c="red", s=25)

        fig.tight_layout()
        plt.show(block=False)
        no = str(input("Continue? "))
        plt.close()
        if no == "n":
            exit(0)
