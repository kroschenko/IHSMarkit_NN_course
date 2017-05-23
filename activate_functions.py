import numpy as np

class Function:

    def apply(self, x):
        return NotImplemented

    def applyD(self, x):
        return NotImplemented

class Logistic(Function):

    def apply(self, x):
        # return .5 * (1 + np.tanh(.5 * x))
        return 1.0 / (1 + np.exp(-x))

    def applyD(self, x):
        return x * (1 - x)

class Tanh_Le(Function):

    def apply(self, x):
        return 1.7159 * np.tanh(2.0/3 * x)

    def applyD(self, x):
        return 2.0/3 * (1.7159 - x*x / 1.7159)

class Tanh(Function):

    def apply(self, x):
        return np.tanh(x)

    def applyD(self, x):
        return 1 - x*x

class Softmax(Function):

    def apply(self, x):
        Zshape = (x.shape[0], 1)
        acts = x - x.max(axis=1).reshape(*Zshape)
        acts = np.exp(acts)
        return acts/acts.sum(axis=1).reshape(*Zshape)

    def applyD(self, x):
        return x * (1 - x)

class ReLU(Function):

    def apply(self, x):
        return np.abs(x)*(x > 0)

    def applyD(self, x):
        return x > 0

class Linear(Function):

    def apply(self, x):
        return x

    def applyD(self, x):
        return np.ones(x.shape)

class Harrington(Function):

    def apply(self, x):
        return np.exp(-np.exp(-x))

    def applyD(self, x):
        return np.exp(-(x + np.exp(-x)))

def get_activate_function(str_act_func):
    if str_act_func == 'logistic':
        return Logistic()
    elif str_act_func == 'softmax':
        return Softmax()
    elif str_act_func == 'tanh':
        return Tanh()
    elif str_act_func == 'relu':
        return ReLU()
    elif str_act_func == 'harrington':
        return Harrington()
    elif str_act_func == 'linear':
        return Linear()
    elif str_act_func == 'tanh_le':
        return Tanh_Le()
    else:
        raise Exception()