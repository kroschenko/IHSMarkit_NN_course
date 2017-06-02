# -*- coding: utf-8 -*-
import numpy as np
from network import Network
from layer import FullyConnectedLayer
from activate_functions import *
from backpropagation import *
import matplotlib.pyplot as plt
import random
import itertools as it

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

def get_ngrams_data_presentation(data):
    ngrams = get_ngrams()
    ngram_data = np.zeros((len(data), len(ngrams)))
    for i in xrange(0, len(data)):
        for j in xrange(0, len(ngrams)):
            if ngrams[j] in data[i]:
                ngram_data[i, j] += 1
    return ngram_data

def get_random_words(data, count):
    word_list = []
    for i in range(0, count):
        _str = data[random.randint(0, len(data))]
        word_list.append(_str)
    return word_list, get_ngrams_data_presentation(word_list)

def search_for_similar_words(words, data, count):
    similar = []
    for word in words:
        similar.append(np.argsort(np.linalg.norm(data - word, axis=1))[0:count])
    return similar

if __name__ == "__main__":
    _dict = load_dictionary('Datasets/dictionary.txt')

    words, words_ngram = get_random_words(_dict, 10)
    data = get_ngrams_data_presentation(_dict)
    net = Network.load_network('nets/network.net')
    output_all_data = net.activate_before_layer(data, 0)
    output_words = net.activate_before_layer(words_ngram, 0)

    similar_words = search_for_similar_words(output_words, output_all_data, 10)
    similar_words = np.array(similar_words).astype('int')
    print similar_words

    for word, similar_indexes in it.izip(words, similar_words):
        print '>' + word
        for s_word_index in similar_indexes:
            print _dict[s_word_index]
