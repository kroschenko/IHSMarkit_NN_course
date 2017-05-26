# -*- coding: utf-8 -*-
import numpy as np
import itertools as it
import random
import matplotlib.pyplot as plt

class WidrowHoffNeuron:
    def __init__(self, rate, Em):
        self.w1 = random.random()
        self.w2 = random.random()
        self.T = random.random()
        self.rate = rate
        self.Em = Em

    def activate(self, sample):
        weightedSum = self.w1 * sample[0] + self.w2 * sample[1] + self.T
        return weightedSum

    def test(self, samples):
        for sample in samples:
            weightedSum = self.w1 * sample[0] + self.w2 * sample[1] + self.T
            print '(' + str(sample[0]) + ', ' + str(sample[1]) + '): ' + str(weightedSum)

    def train(self, samples, targets):
        epochs_count = 0
        isFinish = False
        error_curve = []
        while not isFinish:
            E = 0
            for sample, target in it.izip(samples, targets):
                y = self.activate(sample)
                E += (y - target) * (y - target)
                self.w1 -= self.rate * (y - target) * sample[0]
                self.w2 -= self.rate * (y - target) * sample[1]
                self.T -= self.rate * (y - target)
            epochs_count += 1
            isFinish = E / 2 < self.Em
            error_curve.append(E)
        return error_curve, epochs_count

def plot(error_curve):
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch number', fontsize=18)
    ax.set_ylabel('Error', fontsize=18)
    ax.set_title('Result of learning', fontsize=18)
    plt.plot([x for x in range(0, len(error_curve))], error_curve)
    plt.show()

if __name__ == "__main__":
    samples = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])
    targets = np.array([0.3, 0.6, 0.9])
    neuron = WidrowHoffNeuron(0.2, 1e-5)
    neuron.test(samples)
    error_curve, epochs_count = neuron.train(samples, targets)
    test_samples = np.array([[2.3, 2.4], [2.6, 2.7]])
    neuron.test(samples)
    neuron.test(test_samples)
    print 'Epochs count = ' + str(epochs_count)
    plot(error_curve)

