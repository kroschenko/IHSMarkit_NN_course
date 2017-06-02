import numpy as np
import network
import layer
import backpropagation
import activate_functions
from sklearn.model_selection import train_test_split
from matplotlib import style
import matplotlib.pyplot as plt
RANDOM_SEED = 42


def load_data(path):
    f = open(path)
    data = []
    for _str in f:
        components = _str.rstrip('\n').split()
        data.append(components)
    data = np.array(data).astype('float')
    # order = np.random.permutation(len(data))
    # data = data[order]
    targets = data[:, len(data[0])-1]
    data = data[:, 0:len(data[0]) - 1]
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    print len(data)
    return train_test_split(data, targets, test_size=0.2, random_state=RANDOM_SEED)
    # return data, targets

    # str = f.readline()
    # while f.next():
    #     str =

def MAPE(net, data, targets):
    output = net.activate(data)
    # print output
    _targets = targets.reshape((len(targets), 1))
    mae_error = (np.abs(output - _targets) / _targets).sum()/len(_targets) * 100
    arr = np.hstack((output, _targets))
    # print _targets
    # print output
    return mae_error, arr

def plot_prediction_hist(arr):
    style.use('ggplot')

    # print arr_test
    x =range(0, 40, 2)
    y = arr[0:20, 0]

    # x2 = [5, 8, 10]
    x2 = range(1, 40, 2)
    y2 = arr[0:20, 1]

    predicted = plt.bar(x, y, color='b', align='center', label='predicted')

    actual = plt.bar(x2, y2, color='g', align='center', label='actual')

    plt.title('Boston houses regression task')
    plt.ylabel('Prices')
    plt.xlabel('#')
    plt.legend(handles=[predicted, actual])

    plt.show()


if __name__ == "__main__":
    data = load_data("Datasets/housing.data.txt")
    net = network.Network()
    layer_1 = layer.FullyConnectedLayer(activate_functions.Logistic(), 13, 40)
    layer_2 = layer.FullyConnectedLayer(activate_functions.Linear(), 40, 1)
    net.append_layer(layer_1)
    net.append_layer(layer_2)
    params = backpropagation.Backprop_params(2000, 1e-5, 10, 0.9, False, [0.01, 0.01], 0.00001)
    method = backpropagation.Backpropagation(params, net)
    train_data = data[0]
    test_data = data[1]
    train_labels = data[2]
    test_labels = data[3]
    print len(train_data)
    print len(test_data)

    error_curve = method.train(train_data, train_labels)
    mae_test, arr_test = MAPE(net, test_data, test_labels)
    mae_train, arr_train = MAPE(net, train_data, train_labels)

    print 'MAPE test = ' + str(mae_test)
    print 'MAPE train = ' + str(mae_train)

    plot_prediction_hist(arr_test)
    plot_prediction_hist(arr_train)

