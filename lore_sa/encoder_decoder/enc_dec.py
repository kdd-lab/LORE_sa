from abc import abstractmethod
import numpy as np

__all__ = ["EncDec"]



class EncDec():
    """
    Generic class to implement an encoder/decoder

    It is implemented by different classes, each of which must implements the functions: enc, dec, enc_fit_transform
    the idea is that the user sends the complete record and here only the categorical variables are handled
    """
    def __init__(self,dataset_descriptor):
        self.dataset_descriptor = dataset_descriptor
        self.encoded_features = {}
        self.encoded_descriptor = None

    @abstractmethod
    def encode(self, x: np.array):
        """
        It applies the encoder to the input features

        :param[Numpy array] x: the Dataset containing the features to be encoded
        :param[list] features_to_encode: list of columns of Dataset.df dataframe to be encoded
        """
        return

    @abstractmethod
    def get_encoded_features(self):
        """
        Provides a dictionary with the new encoded features name and the new index
        :return:
        """
        return

    def get_encoded_intervals(self):
        """
        Returns a list of intervals that contains the lower and upper indices of the encoded
        values of features
        """
        return

    @abstractmethod
    def decode(self, x: np.array):
        return


    @abstractmethod
    def decode_target_class(self, x: np.array):
        return

    @abstractmethod
    def encode_target_class(self, param):
        pass

    @abstractmethod
    def decode_rule(self, rule):
        pass