import math
import tensorflow as tf
import torch.nn as tnn
from collections import OrderedDict

class Module(object):
    def __init__(self):
        self.training = True
        self._parameters = []
        self._modules = OrderedDict()

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    def parameters(self):
        return self._parameters

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
            self._parameters += value._parameters
            super(Module, self).__setattr__(name, value)
        else:
            super(Module, self).__setattr__(name, value)

    def __repr__(self):
        my_line = self.__class__.__name__ + '('

        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)

        if len(child_lines) > 0:
            my_line += '\n  ' + '\n  '.join(child_lines) + '\n'

        my_line += ')'
        return my_line

    def __str__(self):
        return self.__repr__()

    def _addindent(s_, numSpaces):
        s = s_.split('\n')
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return tf.nn.relu(x)

    def __repr__(self):
        return 'ReLU()'
