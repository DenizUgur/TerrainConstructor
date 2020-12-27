from dataset import TerrainDataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    DS = TerrainDataset("data/MDRS/data/*.tif")

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
