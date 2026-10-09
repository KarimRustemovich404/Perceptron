"""
Полносвязный (Dense) слой.

Прямой проход для батча X (N x fan_in):
    Z = X W + b        W: (fan_in x units), b: (1 x units)
    A = f(Z)

Обратный проход, получив G = dL/dA:
    dZ = dL/dZ = backward активации (G)
    dW = X^T dZ
    db = сумма dZ по батчу
    dX = dZ W^T        -> передаётся предыдущему слою
"""
import numpy as np

from .activations import get_activation
from .initializers import get_initializer


class DenseLayer:
    def __init__(self, units, activation="sigmoid", weights_initializer="heUniform"):
        self.units = int(units)
        self.activation_name = activation
        self.initializer_name = weights_initializer
        self.activation = get_activation(activation)
        self.W = None
        self.b = None

    def build(self, fan_in, rng):
        init = get_initializer(self.initializer_name)
        self.W = init(fan_in, self.units, rng)
        self.b = np.zeros((1, self.units))

    def forward(self, X):
        self.X = X
        Z = X @ self.W + self.b
        return self.activation.forward(Z)

    def backward(self, grad):
        dZ = self.activation.backward(grad)
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)
        return dZ @ self.W.T

    def update(self, learning_rate):
        # шаг градиентного спуска: theta <- theta - lr * dL/dtheta
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def config(self):
        return {
            "units": self.units,
            "activation": self.activation_name,
            "weights_initializer": self.initializer_name,
        }
