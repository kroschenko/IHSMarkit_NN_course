import numpy as np
import network
import layer
from activate_functions import Logistic
import backpropagation as bpr

def prepareData():
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 0])
    return data, labels

# def plot(error_curve):
#     plt.plot([x for x in range(0, len(error_curve))], error_curve)
#     plt.show()

def test(net, data):
    output = net.activate(data)
    print output

net = network.Network()
layer_1 = layer.FullyConnectedLayer(Logistic(), 2, 2)
layer_2 = layer.FullyConnectedLayer(Logistic(), 2, 1)
net.append_layer(layer_1)
net.append_layer(layer_2)
params = bpr.Backprop_params(30000, 1e-5, 1, 0.9, 0, [0.7, 0.7], 0)
method = bpr.Backpropagation(params, net)
data, labels = prepareData()
method.train(data, labels)
# plot(error_curve)
test(net, data)




