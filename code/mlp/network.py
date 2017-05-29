import pickle

class Network:
    def __init__(self):
        self.layers = []

    def append_layer(self, layer):
        self.layers.append(layer)

    def activate(self, data):
        output = data
        for layer in self.layers:
            output = layer.activate(output)
        return output

    def activate_with_append(self, data):
        output = data
        outputs = []
        outputs_d = []
        outputs.append(output)
        i = 0
        for layer in self.layers:
            output, output_d = layer.activate_with_append(output)
            outputs.append(output)
            outputs_d.append(output_d)
            i += 1
        return output, outputs, outputs_d

    def activate_before_layer(self, data, layer_num):
        output = data
        i = 0
        while i <= layer_num:
            output = self.layers[i].activate(output)
            i += 1
        return output

    @staticmethod
    def save_network(net, path):
        with open(path, 'wb') as f:
            pickle.dump(net, f)

    @staticmethod
    def load_network(path):
        with open(path, 'rd') as f:
            net = pickle.load(f)
        return net
