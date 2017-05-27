# -*- coding: utf-8 -*-
import numpy as np
import itertools as it

class HebbNeuron:
    def __init__(self):
        self.w1 = 0
        self.w2 = 0
        self.T = 0

    def test(self, samples):
        for sample in samples:
            weightedSum = self.w1 * sample[0] + self.w2 * sample[1] + self.T
            if weightedSum > 0:
                y = 1
            else:
                y = -1
            print '(' + str(sample[0]) + ', ' + str(sample[1]) + '): ' + str(y)

    def train(self, samples, targets):
        i = 0
        for sample, target in it.izip(samples, targets):
            i += 1
            self.w1 += sample[0] * target
            self.w2 += sample[1] * target
            self.T += target
            print str(self.w1) + ' - ' + str(self.w2) + ' - ' + str(self.T)

if __name__ == "__main__":
    samples = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    targets = np.array([-1, 1, 1, 1])
    neuron = HebbNeuron()
    neuron.test(samples)
    print '----------------------'
    neuron.train(samples, targets)
    print '----------------------'
    neuron.test(samples)
