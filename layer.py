import numpy as np

class Layer:
    NotImplemented

class FullyConnectedLayer(Layer):
    def __init__(self, act_fun, inputs=None, outputs=None, weights=None, biases=None):
        self.activate_function = act_fun
        if not(inputs is None) and not (outputs is None):
            #initialization types
            # self.weights = np.random.random((inputs, outputs))*0.2 - 0.1
            # self.biases = np.random.random(outputs)*0.2 - 0.1
            self.weights = np.random.randn(inputs, outputs) * 0.1
            self.biases = np.random.randn(outputs) * 0.1
        elif not(weights is None) and not(biases is None):
            self.weights = weights
            self.biases = biases
        else:
            raise Exception()

    def activate(self, data):
        weighted_sums = np.dot(data, self.weights) + self.biases
        return self.activate_function.apply(weighted_sums)

    def activate_with_append(self, data):
        weighted_sum = np.dot(data, self.weights) + self.biases
        act_res = self.activate_function.apply(weighted_sum)
        act_d_res = self.activate_function.applyD(act_res)
        return act_res, act_d_res
