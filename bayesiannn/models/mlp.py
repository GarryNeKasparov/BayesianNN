import numpy as np


class MLP:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        out_func: callable,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.out_func = out_func

        self.w1 = []
        self.w2 = []

    def init_weights(self, w1, w2):
        self.w1 = w1
        self.w2 = w2

    def __call__(self, x, w1, w2):
        z1 = np.dot(x, w1)
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, w2)
        return self.out_func(z2)
