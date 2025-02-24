import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return np.mean((y_true - y_pred) ** 2)

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs
        N = np.prod(y_pred.shape)
        grad_y_pred = (2 / N) * (y_pred - y_true)
        grad_y_true = np.zeros_like(y_true)

        return [grad_y_pred, grad_y_true]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        epsilon = 1e-12  
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  

        loss = -np.sum(y_true * np.log(y_pred), axis=-1) 
        return np.mean(loss)

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs  

        grad_y_pred = -y_true / y_pred  
        grad_y_true = np.zeros_like(y_true)  

        return [grad_y_pred, grad_y_true]
