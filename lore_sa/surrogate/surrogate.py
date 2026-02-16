from abc import ABC, abstractmethod

__all__ = ["Surrogate"]

import numpy as np

from ..dataset import Dataset
from ..encoder_decoder import EncDec


class Surrogate(ABC):
    """
    Abstract base class for interpretable surrogate models.
    
    A surrogate model is an interpretable machine learning model (like a decision tree)
    that approximates the behavior of a complex black box model in a local region around
    a specific instance. LORE uses surrogates to extract interpretable rules that explain
    black box predictions.
    
    The surrogate model is trained on a synthetic neighborhood of instances generated
    around the instance to explain, with labels provided by the black box model. This
    creates a local approximation that is both accurate and interpretable.
    
    Key responsibilities:
    1. Train an interpretable model on the neighborhood
    2. Extract factual rules explaining the prediction
    3. Generate counterfactual rules showing alternative scenarios
    4. Measure fidelity (how well it approximates the black box)
    
    Attributes:
        kind (str): Type of surrogate model (e.g., 'decision_tree', 'supertree')
        preprocessing: Preprocessing method to apply before training
        fidelity (float): Score indicating how well the surrogate approximates 
            the black box (computed during training)
    
    Methods:
        train: Train the surrogate model on neighborhood data
        get_rule: Extract the decision rule for a specific instance
        get_counterfactual_rules: Generate counterfactual rules
    
    Example:
        >>> from lore_sa.surrogate import DecisionTreeSurrogate
        >>> 
        >>> surrogate = DecisionTreeSurrogate()
        >>> surrogate.train(neighborhood_X, neighborhood_y)
        >>> rule = surrogate.get_rule(instance, encoder)
        >>> counterfactuals = surrogate.get_counterfactual_rules(instance, 
        ...                                                       neighborhood_X, 
        ...                                                       neighborhood_y, 
        ...                                                       encoder)
    
    See Also:
        DecisionTreeSurrogate: Concrete implementation using scikit-learn decision trees
    """
    def __init__(self, kind = None, preprocessing =None):
        """
        Initialize the surrogate model.
        
        Args:
            kind (str, optional): Type of surrogate model (e.g., 'decision_tree', 'supertree')
            preprocessing (optional): Preprocessing method to apply to the data before training
        """
        #decision tree, supertree
        self.kind = kind
        #kind of preprocessing to apply
        self.preprocessing = preprocessing

    @abstractmethod
    def train(self, Z, Yb, weights):
        """
        Train the surrogate model on neighborhood data.
        
        This method trains the interpretable surrogate model on a synthetic neighborhood
        of instances, where the labels are provided by the black box model. The goal is
        to create a local approximation that captures the black box's decision boundaries.
        
        Args:
            Z (np.array): Training data in encoded space, shape (n_samples, n_encoded_features).
                These are the synthetic instances generated around the instance to explain.
            Yb (np.array): Target labels from the black box model, shape (n_samples,).
                These are the predictions made by the black box on the neighborhood.
            weights (np.array, optional): Sample weights for training, shape (n_samples,).
                Can be used to emphasize certain instances in the neighborhood.
        
        Note:
            The fidelity of the surrogate is typically computed during training to assess
            how well it approximates the black box model in the local neighborhood.
        """
        pass

    @abstractmethod
    def get_rule(self, x: np.array, encdec: EncDec = None):
        """
        Extract the decision rule for a specific instance.
        
        This method traverses the trained surrogate model to extract the decision rule
        that applies to the given instance. The rule describes the conditions (premises)
        that lead to the predicted class (consequence).
        
        Args:
            x (np.array): Instance to explain, in encoded space, shape (n_encoded_features,)
            encdec (EncDec, optional): Encoder/decoder to convert the rule back to 
                original feature space for interpretability
        
        Returns:
            Rule: Rule object containing premises (conditions) and consequence (prediction)
                that explain why the surrogate (and by extension, the black box) predicts
                a specific class for this instance
        
        Example:
            >>> rule = surrogate.get_rule(encoded_instance, encoder)
            >>> print(rule)
            # Output: IF age > 30 AND income <= 50000 THEN class = 0
        """
        pass

    @abstractmethod
    def get_counterfactual_rules(self, x: np.array, neighborhood_train_X: np.array, neighborhood_train_Y: np.array,
                                 encoder: EncDec = None,
                                 filter_crules=None, constraints: dict = None, unadmittible_features: list = None):
        """
        Generate counterfactual rules showing alternative scenarios.
        
        Counterfactual rules describe what changes to the instance would result in a
        different prediction. They answer "what if" questions like: "What if the age
        was lower? Would the prediction change?"
        
        This method finds paths in the surrogate model that lead to different classes
        and extracts the minimal changes (deltas) needed to reach those predictions.
        
        Args:
            x (np.array): Instance to explain, in encoded space, shape (n_encoded_features,)
            neighborhood_train_X (np.array): Neighborhood instances in encoded space,
                shape (n_samples, n_encoded_features)
            neighborhood_train_Y (np.array): Labels for neighborhood instances from the
                black box, shape (n_samples,)
            encoder (EncDec, optional): Encoder/decoder for converting rules to original space
            filter_crules (optional): Function to filter counterfactual rules
            constraints (dict, optional): Constraints on which features can be changed
            unadmittible_features (list, optional): List of features that cannot be 
                changed (e.g., immutable features like age, gender)
        
        Returns:
            tuple: (counterfactual_rules, deltas) where:
                - counterfactual_rules (list): List of Rule objects for different classes
                - deltas (list): List of lists of Expression objects showing minimal 
                  changes needed for each counterfactual
        
        Example:
            >>> crules, deltas = surrogate.get_counterfactual_rules(
            ...     encoded_instance, neighborhood_X, neighborhood_y, encoder
            ... )
            >>> print(f"Counterfactual: {crules[0]}")
            >>> print(f"Changes needed: {deltas[0]}")
            # Changes needed: [age >= 40, income > 60000]
        """
        pass