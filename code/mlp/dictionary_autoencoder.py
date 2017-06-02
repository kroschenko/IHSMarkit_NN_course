# -*- coding: utf-8 -*-
import numpy as np
from network import Network
from layer import FullyConnectedLayer
from activate_functions import *
from backpropagation import *

def load_dictionary(path):
    input_file = open(path, 'rb')
    _dict = []
    for i in input_file:
        _str = i.rstrip('\n').decode('utf-8')
        if not ('-' in _str) and len(_str) > 1:
            _dict.append(_str)
    return _dict

def get_ngrams():
    alphabet = u"абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    ngrams = []
    for i in range(0, len(alphabet)):
        for j in range(0, len(alphabet)):
            ngrams.append(alphabet[i] + alphabet[j])
    return ngrams

def get_learning_set():
    _dict = load_dictionary('Datasets/dictionary.txt')
    ngrams = get_ngrams()
    data = np.zeros((len(_dict), len(ngrams)))
    for i in xrange(0, len(_dict)):
        for j in xrange(0, len(ngrams)):
            if ngrams[j] in _dict[i]:
                data[i, j] += 1
    return data

if __name__ == "__main__":
    data = get_learning_set()
    
    net = Network()
    layer_1 = FullyConnectedLayer(Logistic(), 1089, 100)
    layer_2 = FullyConnectedLayer(Linear(), 100, 1089)
    net.append_layer(layer_1)
    net.append_layer(layer_2)
    params = Backprop_params(200, 1e-5, 1000, 0.9, False, [0.01, 0.01], 0)
    method = Backpropagation(params, net)

    rnd_index = np.random.permutation(len(data))
    data = data[rnd_index]

    method.train(data, data)

    Network.save_network(net, 'nets/network.net')    






