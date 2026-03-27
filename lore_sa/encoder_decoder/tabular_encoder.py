from lore_sa.encoder_decoder import EncDec, LabelEnc, OneHotEnc
import numpy as np

__all__ = ["EncDec", "TabularEnc","LabelEnc"]
class TabularEnc(EncDec):
    """
    It combines different encoders/decoders over a table dataset
    """
    def __init__(self,descriptor: dict, target_class: str = None):
        super().__init__(descriptor)
        self.target_class = target_class
        self.type='tabular'

    def encode(self, x: np.array):
        """
        Combine one-hot encoding, as first, and label encoding to provide a table encoded.
        :param [Numpy array] x: Array to encode
        :return [Numpy array]: Encoded array
        """
        self.one_hot_enc = OneHotEnc(self.dataset_descriptor)
        one_hot_encoded = self.one_hot_enc.encode(x)
        self.encoded_features.update(self.one_hot_enc.get_encoded_features())

        self.label_enc = LabelEnc(self.one_hot_enc.encoded_descriptor)
        label_encoded = self.label_enc.encode(one_hot_encoded)
        self.encoded_features.update(self.label_enc.get_encoded_features())

        self.encoded_descriptor = self.label_enc.encoded_descriptor
        return np.array([int(n) for n in label_encoded])

    def get_encoded_features(self):
        if self.encoded_features is None:
            raise Exception("You have not run the encoder yet")
        else:
            return self.encoded_features

    def decode(self, x: np.array):
        """
        Decode the instance applyng the onehot and label decode methods
        :param x:
        :return:
        """
        label_enc = LabelEnc(self.dataset_descriptor)
        one_hot_enc = OneHotEnc(self.dataset_descriptor)
        one_hot_decoded = one_hot_enc.decode(x)
        label_decoded = label_enc.decode(one_hot_decoded)
        return label_decoded

    def decode_target_class(self, x: np.array):
        """
        Provides the value of the target class only.
        :param x:
        :return:
        """
        label_dec = LabelEnc({'target':self.dataset_descriptor['target']})
        label_decoded = label_dec.decode_target_class(x)
        return label_decoded