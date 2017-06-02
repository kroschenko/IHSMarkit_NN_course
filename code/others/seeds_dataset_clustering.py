import PCA
import numpy as np
import kmeans
import kohonenmap

def load_dataset(path):
    f = open(path, 'r')
    data = []
    for _str in f:
        _str = _str.rstrip('\n').split(',')
        data.append(_str)
    data = np.array(data).astype('float')
    return data[:, 0:7], data[:, 7]

data, labels = load_dataset("Datasets/seeds.txt")
ind = np.random.permutation(len(labels))
data = data[ind]
labels = labels[ind] - 1

kmeans_method = kmeans.KMeans(data, 3)
clustering = kmeans_method.run()

data = PCA.pca_method_sklearn(data)

PCA.visualise_data(data, clustering)
