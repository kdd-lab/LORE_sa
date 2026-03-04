from abc import abstractmethod
import warnings

import numpy as np

from lore_sa.bbox import AbstractBBox
from lore_sa.dataset import Dataset
from lore_sa.encoder_decoder import EncDec

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

        columns = [None for e in range(len(self.encoder.get_encoded_features().items()))]
        instance = np.zeros(len(columns))
        if from_z is not None:
            instance = from_z  # -1 because the target class is not generated

        for name, feature in self.dataset.descriptor['categorical'].items():
            if self.encoder is not None:
                # feature is encoded, so i need to random generate chunks of one-hot-encoded values

                # finding the vector index of the feature
                indices = [k for k, v in self.encoder.get_encoded_features().items() if v.split("=")[0] == name]
                index_choice = np.random.choice(list(range(len(indices))))

                for i, idx in enumerate(indices):
                    if i == index_choice:
                        instance[idx] = 1
                    else:
                        instance[idx] = 0
                    columns[idx] = self.encoder.get_encoded_features()[idx]


            else:
                # feature is not encoded: random choice among the distinct values of the feature

                instance[feature['index']] = np.random.choice(feature['distinct_values'])
                columns[feature['index']] = name

        for name, feature in self.dataset.descriptor['numeric'].items():
            idx = None
            if self.encoder is not None:
                idx = [k for k, v in self.encoder.get_encoded_features().items() if v == name][0]
            else:
                idx = feature['index']
            columns[idx] = name

            instance[idx] = np.random.uniform(low=feature['min'], high=feature['max'])
        self.columns = columns
        return instance

    def balance_neigh(self, x, Z, num_samples):
        Yb = self.bbox.predict(Z)
        class_counts = np.unique(Yb, return_counts=True)

        if len(class_counts[0]) <= 2:
            ocs = int(np.round(num_samples * self.ocr))
            Z1 = self.__rndgen_not_class(x, ocs, self.bbox.predict(x.reshape(1, -1))[0])
            if len(Z1) > 0:
                Z = np.concatenate((Z, Z1), axis=0)
        else:
            max_cc = np.max(class_counts[1])
            max_cc2 = np.max([cc for cc in class_counts[1] if cc != max_cc])
            if max_cc2 / len(Yb) < self.ocr:
                ocs = int(np.round(num_samples * self.ocr)) - max_cc2
                Z1 = self.__rndgen_not_class(x, ocs, self.bbox.predict(x.reshape(1, -1))[0])
                if len(Z1) > 0:
                    Z = np.concatenate((Z, Z1), axis=0)
        return Z

    def __rndgen_not_class(self, x, num_samples, class_value, max_iter=1000):
        Z = list()
        iter_count = 0
        multi_label = isinstance(class_value, np.ndarray)
        while len(Z) < num_samples:
            z = self.generate_synthetic_instance(x)
            y = self.bbox.predict([z])[0]
            flag = y != class_value if not multi_label else np.all(y != class_value)
            if flag:
                Z.append(z)
            iter_count += 1
            if iter_count >= max_iter:
                break

        Z = np.array(Z)
        return Z

    @abstractmethod
    def generate(self, x: np.array, num_instances: int, descriptor: dict):
        """
        It generates similar instances 

        :param x[dict]: the starting instance from the real dataset
        :param num_instances[int]: the number of instances to generate
        :param descriptor[dict]: data descriptor as generated from a Dataset object
        The list (or range) associated to each key is used to randomly choice an element within the list.
        """
        return

    @abstractmethod
    def check_generated(self, filter_function=None, check_fuction=None):
        """
        It contains the logic to check the requirements for generated data
        """
        raise NotImplementedError("This method is not implemented yet")
        return

    def generate_synthetic_instance(self, from_z=None, mutpb=1.0):

        if from_z is None:
            raise RuntimeError("Missing parameter 'from_z' in generate_synthetic_instance")

        columns = [None for e in range(len(self.encoder.get_encoded_features().items()))]
        instance = np.zeros(len(columns))
        if from_z is not None:
            instance = from_z  # -1 because the target class is not generated

        for name, feature in self.dataset.descriptor['categorical'].items():
            if self.encoder is not None:
                # feature is encoded, so i need to random generate chunks of one-hot-encoded values

                # finding the vector index of the feature
                indices = [k for k, v in self.encoder.get_encoded_features().items() if v.split("=")[0] == name]
                index_choice = np.random.choice(list(range(len(indices))))

                for i, idx in enumerate(indices):
                    if i == index_choice:
                        instance[idx] = 1
                    else:
                        instance[idx] = 0
                    columns[idx] = self.encoder.get_encoded_features()[idx]


            else:
                # feature is not encoded: random choice among the distinct values of the feature

                instance[feature['index']] = np.random.choice(feature['distinct_values'])
                columns[feature['index']] = name

        for name, feature in self.dataset.descriptor['numeric'].items():
            idx = None
            if self.encoder is not None:
                idx = [k for k, v in self.encoder.get_encoded_features().items() if v == name][0]
            else:
                idx = feature['index']
            columns[idx] = name

            instance[idx] = np.random.uniform(low=feature['min'], high=feature['max'])
        self.columns = columns
        return instance

    def balance_neigh(self, x, Z, num_samples):
        Yb = self.bbox.predict(Z)
        class_counts = np.unique(Yb, return_counts=True)

        if len(class_counts[0]) <= 2:
            ocs = int(np.round(num_samples * self.ocr))
            Z1 = self.__rndgen_not_class(x, ocs, self.bbox.predict(x.reshape(1, -1))[0])
            if len(Z1) > 0:
                Z = np.concatenate((Z, Z1), axis=0)
        else:
            max_cc = np.max(class_counts[1])
            max_cc2 = np.max([cc for cc in class_counts[1] if cc != max_cc])
            if max_cc2 / len(Yb) < self.ocr:
                ocs = int(np.round(num_samples * self.ocr)) - max_cc2
                Z1 = self.__rndgen_not_class(x, ocs, self.bbox.predict(x.reshape(1, -1))[0])
                if len(Z1) > 0:
                    Z = np.concatenate((Z, Z1), axis=0)
        return Z

    def __rndgen_not_class(self, x, num_samples, class_value, max_iter=1000):
        Z = list()
        iter_count = 0
        multi_label = isinstance(class_value, np.ndarray)
        while len(Z) < num_samples:
            z = self.generate_synthetic_instance(x)
            y = self.bbox.predict([z])[0]
            flag = y != class_value if not multi_label else np.all(y != class_value)
            if flag:
                Z.append(z)
            iter_count += 1
            if iter_count >= max_iter:
                break

        Z = np.array(Z)
        return Z