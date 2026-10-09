"""
Функции активации.

У каждой активации два метода:
  forward(z)      -> a = f(z), значение запоминается для обратного прохода
  backward(grad)  -> dL/dz, получая на вход dL/da (правило цепочки: dL/dz = dL/da * da/dz)
"""
import numpy as np


class Activation:
    def forward(self, z):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Sigmoid(Activation):
    # f(z) = 1 / (1 + e^{-z}),  f'(z) = f(z) * (1 - f(z))
    def forward(self, z):
        z = np.clip(z, -500, 500)  # защита от переполнения exp
        self.out = 1.0 / (1.0 + np.exp(-z))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1.0 - self.out)


class ReLU(Activation):
    # f(z) = max(0, z),  f'(z) = 1 при z > 0, иначе 0
    def forward(self, z):
        self.z = z
        return np.maximum(0.0, z)

    def backward(self, grad):
        return grad * (self.z > 0)


class Tanh(Activation):
    # f(z) = tanh(z),  f'(z) = 1 - tanh^2(z)
    def forward(self, z):
        self.out = np.tanh(z)
        return self.out

    def backward(self, grad):
        return grad * (1.0 - self.out ** 2)


class Linear(Activation):
    def forward(self, z):
        return z

    def backward(self, grad):
        return grad


class Softmax(Activation):
    """
    s_i = e^{z_i} / sum_j e^{z_j}  — превращает выход сети в распределение вероятностей.

    Якобиан: ds_i/dz_j = s_i * (delta_ij - s_j).
    Произведение якобиана на градиент g = dL/ds:
        dL/dz_j = sum_i g_i * s_i * (delta_ij - s_j) = s_j * (g_j - sum_i g_i * s_i)
    В паре с кросс-энтропией это упрощается до (s - y) / N.
    """
    def forward(self, z):
        z = z - np.max(z, axis=1, keepdims=True)  # численная устойчивость, результат не меняется
        e = np.exp(z)
        self.out = e / np.sum(e, axis=1, keepdims=True)
        return self.out

    def backward(self, grad):
        s = self.out
        return s * (grad - np.sum(grad * s, axis=1, keepdims=True))


ACTIVATIONS = {
    "sigmoid": Sigmoid,
    "relu": ReLU,
    "tanh": Tanh,
    "linear": Linear,
    "softmax": Softmax,
}


def get_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"Неизвестная активация '{name}'. Доступны: {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]()
