import pandas as pd
import numpy as np

from .surrogate import DecisionTreeSurrogate, Surrogate
from .bbox import AbstractBBox
from .dataset import TabularDataset, Dataset
from .encoder_decoder import ColumnTransformerEnc, EncDec
from .neighgen.genetic import GeneticGenerator
from .neighgen.neighborhood_generator import NeighborhoodGenerator
from .neighgen.random import RandomGenerator
from .neighgen.genetic_proba_generator import GeneticProbaGenerator


class Lore(object):
    """
    LOcal Rule-based Explanations (LORE) for black box models.
    
    LORE is a model-agnostic explanation method that provides interpretable explanations 
    for the decisions of black box classifiers. It generates explanations in the form of:
    
    1. A decision rule explaining why the black box made a specific prediction
    2. Counterfactual rules showing what changes would lead to different predictions
    3. Feature importance scores indicating which features were most relevant
    
    The method works by:
    
    1. Generating a synthetic neighborhood around the instance to explain
    2. Training an interpretable surrogate model (decision tree) on this neighborhood
    3. Extracting rules from the surrogate model
    4. Computing counterfactuals that show minimal changes needed for different predictions
    
    For more details, see the paper:
    Guidotti, R., Monreale, A., Ruggieri, S., Pedreschi, D., Turini, F., & Giannotti, F. (2018).
    Local rule-based explanations of black box decision systems. arXiv:1805.10820.
    https://arxiv.org/abs/1805.10820
    
    Attributes:
        bbox (AbstractBBox): The black box model to explain
        descriptor (dict): Dataset descriptor containing feature information
        encoder (EncDec): Encoder/decoder for handling feature transformations
        generator (NeighborhoodGenerator): Generator for creating synthetic neighborhood
        surrogate (Surrogate): Interpretable surrogate model (typically a decision tree)
        class_name (str): Name of the target class variable
        feature_importances (list): Feature importances from the last explanation
    
    Examples:
        >>> from lore_sa import TabularGeneticGeneratorLore
        >>> from lore_sa.dataset import TabularDataset
        >>> from lore_sa.bbox import sklearn_classifier_bbox
        >>> 
        >>> # Load dataset and create black box model
        >>> dataset = TabularDataset.from_csv('data.csv', class_name='target')
        >>> bbox = sklearn_classifier_bbox.sklearnBBox(trained_model)
        >>> 
        >>> # Create LORE explainer
        >>> explainer = TabularGeneticGeneratorLore(bbox, dataset)
        >>> 
        >>> # Explain a single instance
        >>> explanation = explainer.explain_instance(instance)
        >>> print(explanation['rule'])
    """

    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec,
                 generator: NeighborhoodGenerator, surrogate: Surrogate):
        """
        Creates a new instance of the LORE method.
        
        Args:
            bbox (AbstractBBox): The black box model to be explained, wrapped in an 
                AbstractBBox object that provides predict() and predict_proba() methods.
            dataset (Dataset): Dataset object containing the training data and descriptor 
                with information about feature types, ranges, and categorical values.
            encoder (EncDec): Encoder/decoder object for transforming features between 
                original and encoded spaces (e.g., one-hot encoding for categorical features).
            generator (NeighborhoodGenerator): Generator for creating synthetic instances 
                around the instance to explain. Options include RandomGenerator, 
                GeneticGenerator, or GeneticProbaGenerator.
            surrogate (Surrogate): Interpretable surrogate model to approximate the black 
                box locally. Typically a DecisionTreeSurrogate.
        
        Note:
            For convenience, you can use the pre-configured subclasses:
            - TabularRandomGeneratorLore: Uses random sampling for neighborhood generation
            - TabularGeneticGeneratorLore: Uses genetic algorithm for neighborhood generation
            - TabularRandGenGeneratorLore: Uses probabilistic genetic algorithm
        """

        super().__init__()
        self.bbox = bbox
        self.descriptor = dataset.descriptor
        self.encoder = encoder
        self.generator = generator
        self.surrogate = surrogate
        self.class_name = dataset.class_name
        self.feature_importances = None


    def explain(self, x: np.array, num_instances=1000):
        """
        Generate a complete explanation for a single instance.
        
        This method generates a local explanation by:
        1. Encoding the instance into the appropriate feature space
        2. Generating a synthetic neighborhood around the instance
        3. Training a surrogate decision tree on the neighborhood
        4. Extracting factual and counterfactual rules
        5. Computing feature importances
        
        Args:
            x (np.array): A 1D numpy array containing the feature values of the instance 
                to explain. Should NOT include the target class. The order of features 
                should match the dataset's feature order.
            num_instances (int, optional): Number of synthetic instances to generate for 
                the neighborhood. More instances lead to better approximation but slower 
                execution. Default is 1000.
        
        Returns:
            dict: A dictionary containing the explanation with the following keys:
                - 'rule' (dict): The decision rule explaining the black box prediction.
                    Contains 'premise' (list of conditions) and 'consequence' (predicted class).
                - 'counterfactuals' (list): List of counterfactual rules showing what 
                    changes would lead to different predictions.
                - 'fidelity' (float): Fidelity score indicating how well the surrogate 
                    model approximates the black box in the local neighborhood (0 to 1).
                - 'deltas' (list): List of minimal changes (deltas) for each counterfactual,
                    showing what features need to change and by how much.
                - 'counterfactual_samples' (list): Actual synthetic instances that have 
                    different predictions than the original instance.
                - 'counterfactual_predictions' (list): The predicted classes for the 
                    counterfactual samples.
                - 'feature_importances' (list): List of tuples (feature_name, importance) 
                    indicating the importance of each feature in the decision.
        
        Example:
            >>> explanation = lore_explainer.explain(instance, num_instances=1500)
            >>> print(f"Rule: {explanation['rule']}")
            >>> print(f"Fidelity: {explanation['fidelity']:.2f}")
            >>> print(f"Top features: {explanation['feature_importances'][:5]}")
        
        Note:
            The fidelity score measures how well the surrogate decision tree approximates
            the black box model's decisions in the local neighborhood. A fidelity close to
            1.0 indicates the explanation is very reliable.
        """
        # map the single record in input to the encoded space
        [z] = self.encoder.encode([x])
        # generate a neighborhood of instances around the projected instance `z`
        neighbour = self.generator.generate(z.copy(), num_instances, self.descriptor, self.encoder)
        dec_neighbor = self.encoder.decode(neighbour)
        # split neighbor in features and class using train_test_split
        neighb_train_X = dec_neighbor[:, :]
        neighb_train_y = self.bbox.predict(neighb_train_X)
        neighb_train_yb = self.encoder.encode_target_class(neighb_train_y.reshape(-1, 1)).squeeze()

        # train the surrogate model on the neighborhood
        # this surrogate could be another model. I would love to try with apriori 
        # or the modified version of SAME (Single tree Approximation MEthod <3 )
        self.surrogate.train(neighbour, neighb_train_yb)

        # extract the feature importances from the decision tree in self.dt
        if hasattr(self.surrogate, 'dt') and self.surrogate.dt is not None:
            intervals = self.encoder.get_encoded_intervals()
            features_ = self.encoder.encoded_descriptor
            importances = self.surrogate.dt.feature_importances_
            # construct a bitmap from the encoded values `z`
            bm = z.copy()
            # the numerical features takes zero values
            for i, _ in enumerate(features_['numeric']):
                bm[i] = 1;
            # multiply the feature importances with the bitmap array
            importances_ = importances * bm
            feature_importances = []
            for start, end in intervals:
                slice_ = importances_[start:end]
                non_zero = slice_[slice_ != 0]
                if len(non_zero) > 0:
                    feature_importances.append(non_zero[0])
                else:
                    feature_importances.append(0)
            feature_names = [self.encoder.encoded_features[start] for start, _ in intervals ]
            self.feature_importances = list(zip(feature_names, feature_importances))
        else:
            self.feature_importances = None # check if an alternative

        # get the rule for the instance `z`, decode using the encoder class
        rule = self.surrogate.get_rule(z, self.encoder)
        # print('rule', rule)

        crules, deltas = self.surrogate.get_counterfactual_rules(z, neighbour, neighb_train_yb, self.encoder)
        # I wants also the counterfactuals in the original space the so called "no_equal", as well the "equals"
        original_class = self.bbox.predict([x])
        no_equal = [x_c.tolist() for x_c,y_c in zip(dec_neighbor, neighb_train_y) if y_c != original_class]
        actual_class = [y_c for x_c,y_c in zip(dec_neighbor, neighb_train_y) if y_c != original_class]
        return {
            # 'x': x.tolist(),
            'rule': rule.to_dict(),
            'counterfactuals': [c.to_dict() for c in crules],
            'fidelity': self.surrogate.fidelity,
            'deltas': [[dd.to_dict() for dd in d] for d  in deltas],
            'counterfactual_samples': no_equal, # here are the cfs
            'counterfactual_predictions': actual_class,
            'feature_importances': self.feature_importances,
        }



class TabularRandomGeneratorLore(Lore):
    """
    LORE explainer for tabular data using random neighborhood generation.
    
    This is a convenience class that automatically configures LORE with:
    - ColumnTransformerEnc for encoding tabular features
    - RandomGenerator for generating synthetic neighborhoods using random sampling
    - DecisionTreeSurrogate as the interpretable model
    
    The RandomGenerator creates synthetic instances by randomly sampling feature values
    from their valid ranges, with a focus on generating instances close to the instance
    being explained.
    
    Args:
        bbox (AbstractBBox): The black box model to explain, wrapped in an AbstractBBox object.
        dataset (TabularDataset): TabularDataset containing the training data and feature descriptors.
    
    Attributes:
        ocr (float): One-Class Ratio, set to 0.1. Controls the balance between instances 
            with the same class as the original and instances with different classes.
    
    Example:
        >>> from lore_sa import TabularRandomGeneratorLore
        >>> from lore_sa.dataset import TabularDataset
        >>> from lore_sa.bbox import sklearn_classifier_bbox
        >>> 
        >>> dataset = TabularDataset.from_csv('data.csv', class_name='target')
        >>> bbox = sklearn_classifier_bbox.sklearnBBox(model)
        >>> explainer = TabularRandomGeneratorLore(bbox, dataset)
        >>> explanation = explainer.explain_instance(instance)
    
    See Also:
        TabularGeneticGeneratorLore: Uses genetic algorithm for better neighborhood generation
        TabularRandGenGeneratorLore: Uses probabilistic genetic algorithm
    """

    def __init__(self, bbox: AbstractBBox, dataset: TabularDataset):
        """
        Creates a LORE explainer with random neighborhood generation.
        
        Args:
            bbox (AbstractBBox): The black box model to be explained.
            dataset (TabularDataset): Dataset with descriptor information.
        """
        encoder = ColumnTransformerEnc(dataset.descriptor)
        generator = RandomGenerator(bbox, dataset, encoder, 0.1) # the last parameter is the ocr
        surrogate = DecisionTreeSurrogate()

        super().__init__(bbox, dataset, encoder, generator, surrogate)

    def explain_instance(self, x: np.array):
        """
        Explain a single instance from the dataset.
        
        Args:
            x (np.array): Instance to explain as a numpy array (without the target class).
        
        Returns:
            dict: Explanation dictionary with rules, counterfactuals, and feature importances.
        """
        return self.explain(x.values)

class TabularGeneticGeneratorLore(Lore):
    """
    LORE explainer for tabular data using genetic algorithm for neighborhood generation.
    
    This is a convenience class that automatically configures LORE with:
    - ColumnTransformerEnc for encoding tabular features
    - GeneticGenerator for generating synthetic neighborhoods using a genetic algorithm
    - DecisionTreeSurrogate as the interpretable model
    
    The GeneticGenerator uses a genetic algorithm to evolve synthetic instances that are
    similar to the instance being explained but cover different regions of the decision
    space. This typically produces better quality neighborhoods than random sampling,
    leading to more accurate and informative explanations.
    
    The genetic algorithm optimizes two objectives:
    1. Similarity to the instance being explained (controlled by alpha1)
    2. Diversity in predicted classes (controlled by alpha2)
    
    Args:
        bbox (AbstractBBox): The black box model to explain, wrapped in an AbstractBBox object.
        dataset (TabularDataset): TabularDataset containing the training data and feature descriptors.
    
    Attributes:
        ocr (float): One-Class Ratio, set to 0.1. Controls the balance between instances 
            with the same class and instances with different classes.
    
    Example:
        >>> from lore_sa import TabularGeneticGeneratorLore
        >>> from lore_sa.dataset import TabularDataset
        >>> from lore_sa.bbox import sklearn_classifier_bbox
        >>> 
        >>> dataset = TabularDataset.from_csv('data.csv', class_name='target')
        >>> bbox = sklearn_classifier_bbox.sklearnBBox(model)
        >>> explainer = TabularGeneticGeneratorLore(bbox, dataset)
        >>> explanation = explainer.explain_instance(instance)
        >>> 
        >>> # Access the explanation components
        >>> print(f"Factual rule: {explanation['rule']}")
        >>> print(f"Counterfactuals: {explanation['counterfactuals']}")
        >>> print(f"Fidelity: {explanation['fidelity']:.2f}")
    
    See Also:
        TabularRandomGeneratorLore: Simpler but faster random generation
        TabularRandGenGeneratorLore: Probabilistic variant of genetic generation
    """

    def __init__(self, bbox: AbstractBBox, dataset: TabularDataset):
        """
        Creates a LORE explainer with genetic algorithm neighborhood generation.
        
        Args:
            bbox (AbstractBBox): The black box model to be explained.
            dataset (TabularDataset): Dataset with descriptor information.
        """
        encoder = ColumnTransformerEnc(dataset.descriptor)
        generator = GeneticGenerator(bbox, dataset, encoder, 0.1)
        surrogate = DecisionTreeSurrogate()

        super().__init__(bbox, dataset, encoder, generator, surrogate)

    def explain_instance(self, x: np.array):
        """
        Explain a single instance from the dataset.
        
        Args:
            x (np.array): Instance to explain as a numpy array (without the target class).
        
        Returns:
            dict: Explanation dictionary with rules, counterfactuals, and feature importances.
        """
        return self.explain(x.values)
        
class TabularRandGenGeneratorLore(Lore):
    """
    LORE explainer for tabular data using probabilistic genetic algorithm.
    
    This is a convenience class that automatically configures LORE with:
    - ColumnTransformerEnc for encoding tabular features
    - GeneticProbaGenerator for generating synthetic neighborhoods using a probabilistic 
      genetic algorithm
    - DecisionTreeSurrogate as the interpretable model
    
    The GeneticProbaGenerator is a variant of the genetic algorithm that incorporates
    probabilistic elements in the generation process, potentially leading to more diverse
    and representative neighborhoods.
    
    Args:
        bbox (AbstractBBox): The black box model to explain, wrapped in an AbstractBBox object.
        dataset (TabularDataset): TabularDataset containing the training data and feature descriptors.
    
    Attributes:
        ocr (float): One-Class Ratio, set to 0.1. Controls the balance between instances 
            with the same class and instances with different classes.
    
    Example:
        >>> from lore_sa import TabularRandGenGeneratorLore
        >>> from lore_sa.dataset import TabularDataset
        >>> from lore_sa.bbox import sklearn_classifier_bbox
        >>> 
        >>> dataset = TabularDataset.from_csv('data.csv', class_name='target')
        >>> bbox = sklearn_classifier_bbox.sklearnBBox(model)
        >>> explainer = TabularRandGenGeneratorLore(bbox, dataset)
        >>> explanation = explainer.explain_instance(instance)
    
    See Also:
        TabularGeneticGeneratorLore: Standard genetic algorithm variant
        TabularRandomGeneratorLore: Simple random generation
    """
     
    def __init__(self, bbox: AbstractBBox, dataset: TabularDataset):
        """
        Creates a LORE explainer with probabilistic genetic algorithm.
        
        Args:
            bbox (AbstractBBox): The black box model to be explained.
            dataset (TabularDataset): Dataset with descriptor information.
        """
        encoder = ColumnTransformerEnc(dataset.descriptor)
        generator = GeneticProbaGenerator(bbox,
                                            dataset,
                                            encoder,
                                            0.1)
        surrogate = DecisionTreeSurrogate()
        super().__init__(bbox, dataset, encoder, generator, surrogate)

    def explain_instance(self, x:np.array):
        """
        Explain a single instance from the dataset.
        
        Args:
            x (np.array): Instance to explain as a numpy array (without the target class).
        
        Returns:
            dict: Explanation dictionary with rules, counterfactuals, and feature importances.
        """
        return self.explain(x.values)