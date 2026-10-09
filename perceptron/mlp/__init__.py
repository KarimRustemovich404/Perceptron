"""Многослойный персептрон, реализованный с нуля на numpy."""
from .layers import DenseLayer
from .model import Network, createNetwork, fit

__all__ = ["DenseLayer", "Network", "createNetwork", "fit"]
