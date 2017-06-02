__author__ = 'Alex'

import numpy as np
import random as rnd
import math


class KohonenMap:

    def __init__(self, shape, dimension, rate0, sigma0, tau2):
        self.weights = np.random.random((shape[0], shape[1], dimension))
        self.tau2 = tau2
        self.rate0 = rate0
        self.rate = rate0
        self.sigma0 = sigma0
        self.sigma = sigma0
        self.shape = shape
        # self.neurons_count = neurons_count

    def core(self, data, iterationsLimit, changeRate):
        samples_count = len(data)
        iterations = 0
        while iterations < iterationsLimit:
            index = rnd.randint(0, samples_count - 1)
            sample = data[index]
            index = self._define_win_neuron(sample)
            topological_locality = self._topological_locality(index)
            for i in range(0, self.shape[0]):
                for j in range(0, self.shape[1]):
                    self.weights[i, j] += self.rate * topological_locality[i, j] * (sample - self.weights[i, j])
            iterations += 1
            if changeRate:
                self._change_rate(iterations)
            self._change_sigma(iterations)

    def train(self, data):
        self.core(data, 1000, True)
        self.rate = 0.01
        self.core(data, 25000, False)

    def print_clusters(self, data):
        clustering = []
        for sample in data:
            index = self._define_win_neuron(sample)
            clustering.append(index[0] + index[1])
        return clustering
            # print str(index) + ': (' + str(sample[0]) + ', ' + str(sample[1]) + ')'

    def _define_win_neuron(self, sample):
        dist = 1e6
        row = col = -1
        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                if np.linalg.norm(sample-self.weights[i,j]) < dist:
                    dist = np.linalg.norm(sample-self.weights[i, j])
                    row = i
                    col = j
        return [row, col]

        # win_neuron_index = np.linalg.norm(self.weights - sample).argmin()
        # np.indices()
        # return win_neuron_index

    def _topological_locality(self, index):
        distance = np.zeros((self.shape[0], self.shape[1]))
        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                distance[i, j] = np.linalg.norm(index-np.array([i, j]))**2
        return np.exp(-distance/(2*self.sigma**2))

    def _change_sigma(self, n):
        tau1 = 1000.0 / math.log(self.sigma0)
        self.sigma = self.sigma0 * math.exp(-n/tau1)

    def _change_rate(self, n):
        self.rate = self.rate0 * math.exp(-n/self.tau2)

