import random
from abc import abstractmethod
import warnings
import numpy as np

import numpy as np

from ..bbox import AbstractBBox
from ..dataset import Dataset
from ..encoder_decoder import EncDec

warnings.filterwarnings("ignore")

__all__ = ["NeighborhoodGenerator"]

class NeighborhoodGenerator(object):
    """
    Abstract class for Neighborhood generator. It defines the basic functionalities
    to balance the instances of the different classes in the generated data
    """

    @abstractmethod
    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, ocr=0.1):
        self.generated_data = None
        self.bbox = bbox
        self.dataset = dataset
        self.encoder = encoder
        self.ocr = ocr
        self.columns = None
        return

    def generate_synthetic_instance(self, from_z=None, mutpb=1.0):

        if from_z is None:
            raise RuntimeError("Missing parameter 'from_z' in generate_synthetic_instance")

        if self.encoder.type == 'one-hot':
            columns = [None for e in range(len(self.encoder.get_encoded_features().items()))]
            instance = np.zeros(len(columns))
            if from_z is not None:
                instance = from_z # -1 because the target class is not generated


            for name, feature in self.dataset.descriptor['categorical'].items():
                if random.random() < mutpb:
                    if self.encoder is not None: # TO CHECK: it may be that the encoder does not exist?
                        # feature is encoded, so i need to random generate chunks of one-hot-encoded values

                        # finding the vector index of the feature
                        indices = [k for k, v in self.encoder.get_encoded_features().items() if v.split("=")[0] == name]
                        index_choice = np.random.choice(indices)
                        instance[indices[0]:indices[-1]+1] = 0
                        instance[index_choice] = 1
                        # check if the instance within indices has at least one 1
                        if np.sum(instance[indices[0]:indices[-1]+1]) == 0:
                            print(f'Missing value: {name} - {indices}')
                    else:
                        # feature is not encoded: random choice among the distinct values of the feature

                        instance[feature['index']] = np.random.choice(feature['distinct_values'])
                        columns[feature['index']] = name

            for name, feature in self.dataset.descriptor['numeric'].items():
                if random.random() < mutpb:
                    idx = None
                    if self.encoder is not None:
                        idx = [k for k, v in self.encoder.get_encoded_features().items() if v == name][0]
                    else:
                        idx = feature['index']
                    columns[idx] = name

                    instance[idx] = np.random.uniform(low=feature['min'], high=feature['max'])
            self.columns = columns

            return instance
        elif self.encoder.type == 'vae':
            # for vae, the generation is done in the latent space, so we just add a small noise to the input vector
            noise = np.random.normal(0, 0.5, size=from_z.shape)
            instance = from_z + noise
            return instance
        else:
            raise RuntimeError(f"Unsupported encoder type: {self.encoder.type}")


    def balance_neigh(self, z, Z, num_samples):
        X = self.encoder.decode(Z)
        # for i in range(len(X)):
        #     if None in X[i]:
        #         X[i] = self.encoder.decode(z.reshape(1, -1))[0]
        Yb = self.bbox.predict(X)
        x = self.encoder.decode(z[np.newaxis, ...])[0]

        class_counts = np.unique(Yb, return_counts=True)

        if len(class_counts[0]) <= 2:
            ocs = int(np.round(num_samples * self.ocr))
            Z1 = self.__rndgen_not_class(z, ocs, self.bbox.predict(x[np.newaxis, ...])[0])
            if len(Z1) > 0:
                Z = np.concatenate((Z, Z1), axis=0)
        else:
            max_cc = np.max(class_counts[1])
            max_cc2 = np.max([cc for cc in class_counts[1] if cc != max_cc])
            if max_cc2 / len(Yb) < self.ocr:
                ocs = int(np.round(num_samples * self.ocr)) - max_cc2
                Z1 = self.__rndgen_not_class(z, ocs, self.bbox.predict(x[np.newaxis, ...])[0])
                if len(Z1) > 0:
                    Z = np.concatenate((Z, Z1), axis=0)
        return Z

    def __rndgen_not_class(self, z, num_samples, class_value, max_iter=10):
        Z = list()
        iter_count = 0
        multi_label = isinstance(class_value, np.ndarray)
        while len(Z) < num_samples:
            z1 = self.generate_synthetic_instance(z)
            x1 = self.encoder.decode(z1[np.newaxis, ...])[0]
            y = self.bbox.predict(x1[np.newaxis, ...])[0]
            flag = y != class_value if not multi_label else np.all(y != class_value)
            if flag:
                Z.append(z1)
            iter_count += 1
            if iter_count >= max_iter:
                break

        Z = np.array(Z)
        return Z
    
    @abstractmethod
    def generate(self, x: np.array, num_instances: int, descriptor: dict, encoder):
        """
        It generates similar instances 

        :param x[dict]: the starting instance from the real dataset
        :param num_instances[int]: the number of instances to generate
        :param descriptor[dict]: data descriptor as generated from a Dataset object
        The list (or range) associated to each key is used to randomly choice an element within the list.
        """
        raise Exception("ERR: You should implement your own version of the generate() function in the subclass.")

        return

    @abstractmethod
    def check_generated(self, filter_function = None, check_fuction = None):
        """
        It contains the logic to check the requirements for generated data
        """
        raise NotImplementedError("This method is not implemented yet")
        return

