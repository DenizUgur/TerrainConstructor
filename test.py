from dataset import Dataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    DS = Dataset()

    blocks, loss = DS.get_blocks(30)
    print("Data loss after blockify {:.2f}%".format(100 * loss))

    #! Temporarily reduce size of blocks
    blocks = blocks[:30]

    interp_blocks = list(DS.get_interpolated(blocks))

    dtm = interp_blocks[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.contourf(dtm, levels=100, cmap="terrain")
    ax2.contourf(blocks[0], levels=100, cmap="jet")

    fig.tight_layout()
    plt.show()