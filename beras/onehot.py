import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        unique_labels = np.unique(data)
        one_hot_vectors = np.eye(len(unique_labels))
        
        self.onehot_dict = {label: one_hot_vectors[i] for i, label in enumerate(unique_labels)}

    def forward(self, data):
        return np.array([self.onehot_dict[label] for label in data])

    def inverse(self, data):
        return np.array([list(self.onehot_dict.keys())[np.argmax(row)] for row in data])
