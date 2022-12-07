import numpy as np

from mpl_toolkits.mplot3d import Axes3D
# from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
# from sklearn import datasets
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def vis_z_distribution(z):
    X = z.reshape(-1, 256).cpu().detach().numpy()[::10]
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=3)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    # ex_variance = np.var(X_pca, axis=0)
    # ex_variance_ratio = ex_variance / np.sum(ex_variance)

    Xax = X_pca[:, 0]
    Yax = X_pca[:, 1]
    Zax = X_pca[:, 2]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xax, Yax, Zax, s=0.5)

    # cdict = {0: 'red', 1: 'green'}
    # labl = {0: 'Malignant', 1: 'Benign'}
    # marker = {0: '*', 1: 'o'}
    # alpha = {0: .3, 1: .5}
    #
    # fig = plt.figure(figsize=(7, 5))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # fig.patch.set_facecolor('white')
    # for l in np.unique(y):
    #     ix = np.where(y == l)
    #     ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
    #                label=labl[l], marker=marker[l], alpha=alpha[l])
    # for loop ends
    ax.set_xlabel("First Principal Component", fontsize=14)
    ax.set_ylabel("Second Principal Component", fontsize=14)
    ax.set_zlabel("Third Principal Component", fontsize=14)

    ax.legend()
    plt.show()