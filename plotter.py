import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(mat, mask=None, anot=True, save_path=None):
    if mask is None:
        mask = np.ones_like(mat)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)

    xlabel = [str(num + 1) for num in range(mat.shape[0])]
    ylabel = [str(num + 1) for num in range(mat.shape[1])]

    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(xlabel, fontdict={'fontsize': 20})
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(ylabel, fontdict={'fontsize': 20})

    norm = mpl.colors.Normalize(vmin=-np.max(mat), vmax=np.max(mat))
    # norm = mpl.colors.Normalize(vmin=0, vmax=np.max(mat))
    im = ax.imshow(mat * mask, cmap=plt.cm.bwr, norm=norm)
    # im = ax.imshow(mat * mask, cmap=plt.cm.Reds, norm=norm)
    if anot:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                tx = ax.text(j, i, np.around(mat[i, j] * mask[i, j], 2), ha='center', va='center')

    cb = plt.colorbar(im, shrink=0.8)
    cb.ax.tick_params(labelsize=24)
    if save_path:
        plt.savefig(save_path)
    
    # plt.show()
    plt.close()
