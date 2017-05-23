import numpy as np

def testing2(net, data, labels):
    output = net.activate_cpu(data)
    answer = output.argmax(1)
    percentage = (answer == labels).sum() / float(data.shape[0])  * 100
    labels = unzip(labels, 3)
    error = ((output - labels)**2).sum()
    print error
    return percentage

def unzip(value, class_num):
    length = len(value)
    unzip_form = np.zeros((length, class_num))
    i = 0
    for row in value:
        unzip_form[i, row] = 1
        i += 1
    return unzip_form

class Backprop_params:
    def __init__(self, max_epochs, min_error, batch_size, momentum, zip, rates, weight_loss):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.zip = zip
        self.rates = rates
        self.weight_loss = weight_loss
        self.min_error = min_error

    # def __init__(self, config):
    #     self.max_epochs = config['training']['max_epochs']
    #     # self.rate = config['training']['rate']
    #     self.batch_size = config['training']['batch_size']
    #     self.momentum = config['training']['momentum']
    #     self.weight_loss = config['training']['weight_loss']
    #     self.zip = config['data']['zip']
    #     self.rates = config['training']['rates']

class Backpropagation:

    def __init__(self, params, net):
        self.params = params
        self.net = net

    def train(self, data, labels):
        errors_serie = []
        if self.params.zip:
            etalon = Backpropagation.unzip(labels, self.net.layers[len(self.net.layers)-1].weights.shape[1])
        else:
            etalon = labels
        batch_size = self.params.batch_size
        i = 0
        #weights and biases matrix updates initialization
        weights_updates = []
        biases_updates = []
        for layer in self.net.layers:
            weights_updates.append(np.zeros(layer.weights.shape))
            biases_updates.append(np.zeros(layer.biases.shape))
        isFinish = True
        while isFinish:
            k = 0
            error = 0
            while k < len(data):
                data_batch = np.array(data[k:k+batch_size])
                values_batch = np.array(etalon[k:k+batch_size])
                output, outputs, derivatives = self.net.activate_with_append(data_batch)
                # derivatives.pop()
                errors = output - values_batch.reshape(output.shape)
                gradients = self.back_propagation(errors, derivatives)
                self.change_weights(gradients, outputs, weights_updates, biases_updates)
                k += batch_size
                error += (errors * errors).sum()
            i += 1
            print str(i) + " epoch is complete... error is " + str(error)
            errors_serie.append(error)
            isFinish = i < self.params.max_epochs and error > self.params.min_error
        return errors_serie

    @staticmethod
    def unzip(value, class_num):
        length = len(value)
        unzip_form = np.zeros((length, class_num))
        i = 0
        for row in value:
            unzip_form[i, row] = 1
            i += 1
        return unzip_form

    def back_propagation(self, errors, derivatives):
        gradients = []
        gradients.append(errors*derivatives.pop())
        # gradients.append(errors)
        for i in range(len(self.net.layers)-1, 0, -1):
            gradients.append(np.dot(gradients[len(gradients)-1], self.net.layers[i].weights.T) * derivatives.pop())
        gradients.reverse()
        return gradients

    def change_weights(self, gradients, outputs, weights_updates, biases_updates):
        samples_count = outputs[0].shape[0]
        i = 0
        for layer in self.net.layers:
            weights_updates[i] *= self.params.momentum
            biases_updates[i] *= self.params.momentum
            weights_updates[i] -= self.params.rates[i] * (np.dot(outputs[i].T, gradients[i]) / samples_count - self.params.weight_loss * layer.weights)
            biases_updates[i] -= self.params.rates[i] / samples_count * gradients[i].sum(0)
            layer.weights += weights_updates[i]
            layer.biases += biases_updates[i]
            i += 1