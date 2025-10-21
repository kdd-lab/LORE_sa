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
    Abstract base class for generating synthetic neighborhoods.
    
    A neighborhood generator creates synthetic instances around a specific instance
    to be explained. These synthetic instances are used to train a local surrogate
    model that approximates the black box's behavior in that region.
    
    The key challenge is generating a diverse neighborhood that:
    1. Covers the local decision boundary
    2. Includes instances with both the same and different predicted classes
    3. Is similar enough to the original instance to provide a local explanation
    
    Different generators use different strategies:
    - RandomGenerator: Pure random sampling within feature ranges
    - GeneticGenerator: Genetic algorithm to evolve good neighborhoods
    - GeneticProbaGenerator: Probabilistic variant of genetic generation
    
    Attributes:
        bbox (AbstractBBox): The black box model to explain
        dataset (Dataset): Dataset with feature descriptors
        encoder (EncDec): Encoder/decoder for feature transformations
        ocr (float): One-Class Ratio - controls the balance between instances with
            the same class (1-ocr) and different classes (ocr) in the neighborhood
        generated_data: The generated neighborhood instances
        columns: Feature names for the generated instances
    
    Methods:
        generate: Generate a synthetic neighborhood
        generate_synthetic_instance: Generate a single synthetic instance
        balance_neigh: Balance the class distribution in the neighborhood
    
    Example:
        >>> from lore_sa.neighgen import GeneticGenerator
        >>> 
        >>> generator = GeneticGenerator(bbox, dataset, encoder, ocr=0.1)
        >>> neighborhood = generator.generate(encoded_instance, 
        ...                                   num_instances=1000, 
        ...                                   descriptor=dataset.descriptor,
        ...                                   encoder=encoder)
    
    See Also:
        RandomGenerator: Simple random sampling
        GeneticGenerator: Genetic algorithm-based generation
        GeneticProbaGenerator: Probabilistic genetic generation
    """

    @abstractmethod
    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, ocr=0.1):
        """
        Initialize the neighborhood generator.
        
        Args:
            bbox (AbstractBBox): The black box model to explain
            dataset (Dataset): Dataset containing feature descriptors
            encoder (EncDec): Encoder/decoder for feature transformations
            ocr (float, optional): One-Class Ratio, controls the balance between
                instances with the same class and different classes. Default is 0.1,
                meaning 90% of instances will have the same predicted class as the
                original instance, and 10% will have different classes.
        """
        self.generated_data = None
        self.bbox = bbox
        self.dataset = dataset
        self.encoder = encoder
        self.ocr = ocr
        self.columns = None
        return

    def generate_synthetic_instance(self, from_z=None, mutpb=1.0):
        """
        Generate a single synthetic instance.
        
        This method creates one synthetic instance by randomly sampling or mutating
        feature values. For categorical features, it randomly selects from valid values.
        For numerical features, it samples from the feature's range.
        
        Args:
            from_z (np.array, optional): Starting instance in encoded space to mutate.
                If None, generates a completely random instance. If provided, features
                are mutated with probability mutpb.
            mutpb (float, optional): Mutation probability for each feature (0 to 1).
                Only used when from_z is provided. Default is 1.0 (mutate all features).
        
        Returns:
            np.array: A single synthetic instance in encoded space, shape (n_encoded_features,)
        
        Note:
            The method respects feature types and valid ranges from the dataset descriptor.
            For categorical features, it ensures the one-hot encoding constraint (exactly
            one category is active).
        """

        if from_z is None:
            raise RuntimeError("Missing parameter 'from_z' in generate_synthetic_instance")

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

    def balance_neigh(self, z, Z, num_samples):
        X = self.encoder.decode(Z)
        for i in range(len(X)):
            if None in X[i]:
                X[i] = self.encoder.decode(z.reshape(1, -1))[0]
        Yb = self.bbox.predict(X)
        x = self.encoder.decode(z.reshape(1, -1))[0]

        class_counts = np.unique(Yb, return_counts=True)

        if len(class_counts[0]) <= 2:
            ocs = int(np.round(num_samples * self.ocr))
            Z1 = self.__rndgen_not_class(z, ocs, self.bbox.predict(x.reshape(1, -1))[0])
            if len(Z1) > 0:
                Z = np.concatenate((Z, Z1), axis=0)
        else:
            max_cc = np.max(class_counts[1])
            max_cc2 = np.max([cc for cc in class_counts[1] if cc != max_cc])
            if max_cc2 / len(Yb) < self.ocr:
                ocs = int(np.round(num_samples * self.ocr)) - max_cc2
                Z1 = self.__rndgen_not_class(z, ocs, self.bbox.predict(x.reshape(1, -1))[0])
                if len(Z1) > 0:
                    Z = np.concatenate((Z, Z1), axis=0)
        return Z

    def __rndgen_not_class(self, z, num_samples, class_value, max_iter=1000):
        Z = list()
        iter_count = 0
        multi_label = isinstance(class_value, np.ndarray)
        while len(Z) < num_samples:
            z1 = self.generate_synthetic_instance(z)
            x1 = self.encoder.decode(z1.reshape(1, -1))[0]
            y = self.bbox.predict([x1])[0]
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

