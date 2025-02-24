import numpy as np

from beras.core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        ## TODO: Compute and return the categorical accuracy of your model 
        ## given the output probabilities and true labels. 
        ## HINT: Argmax + boolean mask via '=='
        predicted_class = np.argmax(probs, axis=-1)
        true_class = np.argmax(labels, axis=-1)

        accuracy = np.mean(predicted_class == true_class)
        return accuracy
