from dataset import TerrainDataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    TDS = TerrainDataset("data/MDRS/data/*.tif", dataset_type="train")

    fig, (r1, r2) = plt.subplots(2, 5, figsize=(25, 10))

    for i, ax in enumerate(r1):
        (_, _), y = TDS[0 + i]
        ax.contourf(y.squeeze(0).numpy(), levels=100, cmap="terrain")

    for i, ax in enumerate(r2):
        (_, _), y = TDS[5 + i]
        ax.contourf(y.squeeze(0).numpy(), levels=100, cmap="terrain")

    fig.tight_layout()
    plt.show()
