import numpy as np
from dataset import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from timeit import default_timer as dt


def main(index):
    global Y
    start = dt()
    DS.prepare()
    print("Prepared in {:.2f} seconds".format(dt() - start))

    start = dt()
    X, Y, O = DS.get(index=index, return_observer=True)
    print("Got a sample in {:.2f} seconds".format(dt() - start))

    # * PLOT
    fig = plt.figure(figsize=(32, 18))
    ax = plt.axes(projection="3d")
    lin_x = np.arange(0, X.shape[0], 1)
    lin_y = np.arange(0, X.shape[1], 1)
    x, y = np.meshgrid(lin_x, lin_y)

    colors = np.empty(X.shape, dtype=str)
    colors[np.isnan(X)] = 'r'
    colors[~np.isnan(X)] = 'y'

    ax.plot_surface(x, y, X, rstride=5, cstride=5)
    ax.scatter(O[0], O[1], O[2], s=50, color="red")

    fig.tight_layout()
    plt.show(block=False)


if __name__ == "__main__":
    DS = Dataset()

    i = 0
    while 1:
        print(i)
        main(i)
        no = str(input("Continue? "))
        plt.close()
        if no == "n":
            exit(0)
        i += 1
