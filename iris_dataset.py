import numpy as np
from network import Network
from layer import FullyConnectedLayer
from activate_functions import Logistic
from activate_functions import Linear
from backpropagation import Backpropagation
from backpropagation import Backprop_params
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
RANDOM_SEED = 42

def prepareData():
    irises_dataset = datasets.load_iris()
    data = irises_dataset['data']
    labels = irises_dataset['target']
    # order = np.random.permutation(len(data))
    # data = data[order]
    # labels = labels[order]
    return train_test_split(data, labels, test_size=0.33, random_state=RANDOM_SEED)

def plot(error_curve):
    plt.plot([x for x in range(0, len(error_curve))], error_curve)
    plt.show()

def testing2(net, data, labels):
    output = net.activate(data)
    answer = output.argmax(1)
    percentage = (answer == labels).sum() / float(data.shape[0])  * 100
    # labels = unzip(labels, 3)
    # error = ((output - labels)**2).sum()
    # print error
    return percentage

net = Network()
layer_1 = FullyConnectedLayer(Logistic(), 4, 256)
# layer_2 = FullyConnectedLayer(Logistic(), 2, 2)
layer_3 = FullyConnectedLayer(Logistic(), 256, 3)
net.append_layer(layer_1)
# net.append_layer(layer_2)
net.append_layer(layer_3)
params = Backprop_params(100, 1e-5, 1, 0.9, True, [0.01, 0.01], 0)
method = Backpropagation(params, net)
data_all = prepareData()
train_data = data_all[0]
test_data = data_all[1]
train_labels = data_all[2]
test_labels = data_all[3]


error_curve = method.train(train_data, train_labels)
plot(error_curve)
print testing2(net, train_data, train_labels)
print testing2(net, test_data, test_labels)



