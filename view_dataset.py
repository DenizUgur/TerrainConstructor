from dataset import TerrainDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    TDS = TerrainDataset("data/MDRS/data/*.tif")
    DL = DataLoader(dataset=TDS, batch_size=1, num_workers=0)

    #! CNN
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    count = 0
    for i, ((x, _), y) in enumerate(DL):

        data = x.to(device)
        target = y.to(device)

        print(target)
        print(target.shape)

        optimizer.zero_grad()
        output = model(data)
        print("data.shape\t", data.shape)
        print("target.shape\t", target.shape)
        print("output.shape\t", output.shape)

        loss = criterion(output, target.squeeze(1).long())
        loss.backward()
        optimizer.step()

        _, prediction = torch.max(output.data, 1)
        npp = prediction.squeeze(0).cpu().numpy()

        count += 1
        if count == 5:
            break

    print(loss)
    plt.contourf(target.cpu().squeeze(1).numpy(), levels=100)
    plt.show()
        

    for (x, (ox, oy, _)), y in TDS:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.contourf(x.squeeze(0).numpy(), levels=100)
        ax2.contourf(y.squeeze(0).numpy(), levels=100)

        ax1.scatter(ox, oy, c="red", s=25)
        ax2.scatter(ox, oy, c="red", s=25)

        fig.tight_layout()
        plt.show(block=False)
        no = str(input("Continue? "))
        plt.close()
        if no == "n":
            exit(0)