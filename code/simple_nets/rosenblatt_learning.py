# -*- coding: utf-8 -*-
import numpy as np
import itertools as it
import random
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'family' : 'serif',
        'weight' : 'normal',
        'size' : 14}
rc('font', **font)

class RosenblattNeuron:
    def __init__(self, rate):
        self.w1 = random.random()
        self.w2 = random.random()
        self.T = random.random()
        self.rate = rate

    def activate(self, sample):
        weightedSum = self.w1 * sample[0] + self.w2 * sample[1] + self.T
        y = self.thresActivateFunction(weightedSum)
        return y

    def thresActivateFunction(self, x):
        if x < 0:
            return -1
        else:
            return 1

    def test(self, samples):
        for sample in samples:
            weightedSum = self.w1 * sample[0] + self.w2 * sample[1] + self.T
            y = self.thresActivateFunction(weightedSum)
            print '(' + str(sample[0]) + ', ' + str(sample[1]) + '): ' + str(y)

    def train(self, samples, targets):
        isFinish = False
        epochs = 0
        print str(self.w1) + ' - ' + str(self.w2) + ' - ' + str(self.T)
        while not isFinish:
            isFinish = True
            for sample, target in it.izip(samples, targets):
                y = self.activate(sample)
                if y != target:
                    isFinish = False
                    self.w1 += self.rate * sample[0] * target
                    self.w2 += self.rate * sample[1] * target
                    self.T += self.rate * target
            epochs += 1
            print str(self.w1) + ' - ' + str(self.w2) + ' - ' + str(self.T)
        print 'epochs = ' + str(epochs)

def plot(neuron):
    fig, ax = plt.subplots()
    rc('text', usetex=True)
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_xlabel('$w_1$', fontsize=18)
    ax.set_ylabel('$w_2$', fontsize=18)
    x = []
    y = []
    i = -2
    while i < 2:
        x.append(i)
        y.append((- neuron.w1 * i - neuron.T) / neuron.w2)
        i += 0.1
    plt.plot(x, y)
    plt.plot([-1, 1, 1, -1], [1, 1, -1, -1], 'o')
    plt.show()

if __name__ == "__main__":
    samples = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    targets = np.array([-1, -1, -1, 1])
    neuron = RosenblattNeuron(0.01)
    neuron.test(samples)
    neuron.train(samples, targets)
    neuron.test(samples)
    plot(neuron)
