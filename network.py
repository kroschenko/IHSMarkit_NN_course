import numpy as np
import layer
import math
import itertools as it
import activate_functions

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
