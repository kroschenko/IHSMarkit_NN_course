# -*- coding: utf-8 -*-
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from matplotlib import rc

# rc('font', **{
# 'family': 'DejaVu Sans',
# 'weight': 'normal'
# })

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# font = {'family': 'Verdana',
#         'weight': 'normal'}
# rc('font', **font)
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
# font = {'family': 'serif', 'weight': 'bold'}
# rc('font', **font)

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size' : 18}
#
rc('font', **font)

class LinearNeuron:
    def __init__(self):
        self.w1 = 0
        self.w2 = 0
        self.T = 0

    def run(self, samples):
        for sample in samples:
            weightedSum = self.w1 * sample[0] + self.w2 * sample[1] + self.T
            if weightedSum > 0:
                y = 1
            else:
                y = -1
            # print '(' + str(sample[0]) + ', ' + str(sample[1]) + '): ' + str(y)

    def train(self, samples, targets):
        i = 0
        print str(i) + ' epoch: ' + 'params = (' + str(self.w1) + ', ' + str(self.w2) + ', ' + str(self.T) + ')'
        for sample, target in it.izip(samples, targets):
            i += 1
            self.w1 += sample[0] * target
            self.w2 += sample[1] * target
            self.T += target
            print str(i) + ' epoch: ' + 'params = (' + str(self.w1) + ', ' + str(self.w2) + ', ' + str(self.T) + ')'

if __name__ == "__main__":
    samples = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    targets = np.array([-1, 1, 1, 1])
    neuron = LinearNeuron()
    neuron.run(samples)
    neuron.train(samples, targets)
    neuron.run(samples)
    x = np.linspace(0.2, 10, 100)
    fig, ax = plt.subplots()
    rc('text', usetex=True)
    rc('text.latex', unicode=True)

    # \usepackage[T2A]
    # {fontenc}
    # \usepackage[utf8]
    # {inputenc}
    # \usepackage[russian, english]
    # {babel}

    rc('text.latex', preamble='\usepackage[T2A]{fontenc} \usepackage[russian]{babel}')
    # plt.rc('font', family='Verdana')
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_xlabel('\\textbf{w_1}', fontsize=22)
    ax.set_ylabel('w_2', fontsize=22)
    ax.set_title(u'Результат обучения нейрона', fontsize=22)

    plt.plot([-3, 2], [2, -3], label=u'Разделяющая прямая')
    plt.plot([-1, 1, 1, -1], [1, 1, -1, -1], 'o', label=u'Значения функции <<ИЛИ>>')
    plt.legend(bbox_to_anchor=(0.3, 0), loc=8, borderaxespad=0., numpoints=1)
    # plt.legend(numpoints=1)
    plt.show()
