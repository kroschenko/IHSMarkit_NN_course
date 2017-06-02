import PCA
import numpy as np
import kmeans
import kohonenmap

def load_dataset(path):
    f = open(path, 'r')
    data = []
    for _str in f:
        # _str = _str.rstrip(['\n', '\t']).split(' ')
        _str = _str.rstrip('\n').split(',')
        data.append(_str)
    data = np.array(data).astype('float')

    return data[:, 0:7], data[:, 7]

data, labels = load_dataset("Datasets/seeds.txt")
ind = np.random.permutation(len(labels))
data = data[ind]
labels = labels[ind] - 1

# data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

# kmeans_method = kmeans.KMeans(data, 3)
# clustering = kmeans_method.run()

# data = PCA.pca_method_sklearn(data)

map = kohonenmap.KohonenMap((3, 1), 7, 0.1, 2., 1000.)

map.train(data)
labels1 = map.print_clusters(data)

data = PCA.pca_method_sklearn(data)
PCA.visualise_data(data, labels1)

PCA.visualise_data(data, labels)

# PCA.visualise_data(data, clustering)
# PCA.visualise_data(data, labels)
# kmeans_method = kmeans.KMeans(data, 3)
# clustering = kmeans_method.run()
# print clustering

# PCA.visualise_data(data, clustering)