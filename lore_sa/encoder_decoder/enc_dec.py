from abc import abstractmethod
<<<<<<< HEAD
from lore_sa.dataset.dataset import TabularDataset

import pandas as pd
=======
from lore_sa.dataset.tabular_dataset import TabularDataset
>>>>>>> dfa71cf528e505c88e80bbe6f7cb96f1c9477049

__all__ = ["EncDec"]
class EncDec():
    """
    Generic class to implement an encoder/decoder

    It is implemented by different classes, each of which must implements the functions: enc, dec, enc_fit_transform
    the idea is that the user sends the complete record and here only the categorical variables are handled
    """
    def __init__(self,):
        self.dataset_encoded = None
        self.original_features = None
        self.original_data = None
        self.encoded_features = None
        self.original_features_encoded = None
        

    @abstractmethod
    def encode(self, x: TabularDataset, features_to_encode):
        """
        It applies the encoder to the input features

        :param[TabularDataset] x: the Dataset containing the features to be encoded
        :param[list] features_to_encode: list of columns of Dataset.df dataframe to be encoded
        """
        return

    @abstractmethod
    def decode(self, x: TabularDataset, kwargs=None):
        return
