'''
Python version: 3.9
Submitters: Yishay Seroussi 305027948, Samuel Bismuth 342533064.
'''


# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
import repackage
repackage.up()
from data import split_data


def k_means_clustering(train_x, test_x, train_y, test_y):
    kmeans = KMeans(n_clusters=10, random_state=0)
    clusters = kmeans.fit_predict(train_x.tolist())
    kmeans.cluster_centers_.shape
    fig, ax = plt.subplots(2, 5, figsize=(8, 3))
    centers = kmeans.cluster_centers_.reshape(10, 28, 28)
    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    labels = np.zeros_like(clusters)
    for i in range(10):
        mask = (clusters == i)
        labels[mask] = mode(train_y[mask])[0]
    plt.show()
    print('k_means_clustering accuracy: {0}'.format(accuracy_score(train_y, labels)))


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = split_data()
    k_means_clustering(train_x, test_x, train_y, test_y)