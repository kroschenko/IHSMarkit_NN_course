import numpy as np
import random as rnd


class KMeans:
    def __init__(self, data, count_clusters):
        count_samples = len(data)
        self.count_clusters = count_clusters
        self.data = data
        self.clustering = np.zeros(count_samples).astype('int')
        self._init_clustering()

    def _init_clustering(self):
        count_samples = len(self.data)
        for i in range(0, count_samples):
            if i < self.count_clusters:
                self.clustering[i] = i
            else:
                self.clustering[i] = int(self.count_clusters * rnd.random())

    def run(self):
        iterations = 0
        while True:
            means = self._update_means()
            centroids = self._compute_centroids(means)
            iterations += 1
            if not self._assign_samples_to_clusters(centroids):
                break
        return self.clustering

    def _update_means(self):
        count_components = len(self.data[0])
        count_samples_in_clusters = np.zeros(self.count_clusters)
        means = np.zeros((self.count_clusters, count_components))
        i = 0
        for elem in self.clustering:
            count_samples_in_clusters[elem] += 1
            means[elem] += self.data[i]
            i += 1
        for i in range(0, self.count_clusters):
            means[i] /= count_samples_in_clusters[i]
        return means

    def _compute_centroids(self, means):
        count_samples = len(self.data)
        count_components = len(self.data[0])
        centroids = np.zeros((self.count_clusters, count_components))
        distance = np.zeros(self.count_clusters)
        centroids_index = np.zeros(self.count_clusters).astype('int')
        for i in range(0, self.count_clusters):
            distance[i] = float('inf')
            centroids_index[i] = -1
        for i in range(0, count_samples):
            num = self.clustering[i]
            current_distance = np.linalg.norm(self.data[i] - means[num])
            if current_distance < distance[num]:
                distance[num] = current_distance
                centroids_index[num] = i
        for i in range(0, self.count_clusters):
            centroids[i] = self.data[centroids_index[i]]
        return centroids

    def _assign_samples_to_clusters(self, centroids):
        change = False
        i = 0
        for sample in self.data:
            distances = np.linalg.norm(sample - centroids, axis=1)
            num = distances.argmin()
            if num != self.clustering[i]:
                self.clustering[i] = num
                change = True
            i += 1
        return change

