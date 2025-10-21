from abc import abstractmethod
import numpy as np

__all__ = ["EncDec"]
class EncDec():
    """
    Abstract base class for encoding and decoding features.
    
    The EncDec class provides an interface for transforming features between their
    original representation and an encoded representation suitable for machine learning
    models. This is particularly important for:
    
    - Categorical features: Converting to one-hot encoding or ordinal encoding
    - Numerical features: Scaling or normalization
    - Feature engineering: Creating derived features
    
    The encoder is used by LORE to:
    1. Encode instances before generating neighborhoods
    2. Decode synthetic instances back to original feature space
    3. Decode rules to make them interpretable
    
    Attributes:
        dataset_descriptor (dict): Descriptor containing information about feature types,
            ranges, and categorical values
        encoded_features (dict): Mapping from encoded feature indices to feature names
        encoded_descriptor (dict): Descriptor for the encoded feature space
    
    Methods:
        encode: Transform features to encoded representation
        decode: Transform encoded features back to original representation
        encode_target_class: Encode target class labels
        decode_target_class: Decode target class labels
        get_encoded_features: Get mapping of encoded features
        get_encoded_intervals: Get index intervals for encoded features
    
    Example:
        >>> from lore_sa.encoder_decoder import ColumnTransformerEnc
        >>> 
        >>> # Create encoder from dataset descriptor
        >>> encoder = ColumnTransformerEnc(dataset.descriptor)
        >>> 
        >>> # Encode a sample
        >>> encoded = encoder.encode([sample])
        >>> 
        >>> # Decode back to original space
        >>> decoded = encoder.decode(encoded)
    
    See Also:
        ColumnTransformerEnc: Concrete implementation using sklearn's ColumnTransformer
    """
    def __init__(self,dataset_descriptor):
        """
        Initialize the encoder/decoder.
        
        Args:
            dataset_descriptor (dict): Dictionary containing feature information including
                'numeric', 'categorical', and 'ordinal' feature descriptors
        """
        self.dataset_descriptor = dataset_descriptor
        self.encoded_features = {}
        self.encoded_descriptor = None

    @abstractmethod
    def encode(self, x: np.array):
        """
        Transform features from original to encoded representation.
        
        This method applies the encoding transformation to convert features from their
        original space (e.g., with categorical labels) to an encoded space suitable for
        machine learning (e.g., with one-hot encoded categorical features).
        
        Args:
            x (np.array): Array of shape (n_samples, n_features) containing samples
                in the original feature space
        
        Returns:
            np.array: Encoded array of shape (n_samples, n_encoded_features) where
                n_encoded_features may be larger than n_features due to one-hot encoding
        
        Example:
            >>> # Original: [['red', 25], ['blue', 30]]
            >>> # Encoded:  [[1, 0, 0, 25], [0, 1, 0, 30]]  # one-hot for color
            >>> encoded = encoder.encode(original_data)
        """
        return

    @abstractmethod
    def get_encoded_features(self):
        """
        Get a mapping of encoded feature indices to feature names.
        
        Returns:
            dict: Dictionary mapping encoded feature indices to descriptive names.
                For one-hot encoded features, names include the category value 
                (e.g., 'color=red', 'color=blue').
        
        Example:
            >>> features = encoder.get_encoded_features()
            >>> # {0: 'age', 1: 'color=red', 2: 'color=blue', 3: 'color=green'}
        """
        return

    def get_encoded_intervals(self):
        """
        Get index intervals for each original feature in the encoded space.
        
        This method returns a list of (start, end) tuples indicating the range of
        encoded indices that correspond to each original feature. This is useful
        when an original categorical feature is one-hot encoded into multiple columns.
        
        Returns:
            list: List of (start_idx, end_idx) tuples, one for each original feature.
                For numerical features, start_idx == end_idx. For one-hot encoded
                categorical features, the interval spans multiple indices.
        
        Example:
            >>> intervals = encoder.get_encoded_intervals()
            >>> # [(0, 1), (1, 4), (4, 5)]  # age (1 col), color (3 cols), income (1 col)
        """
        return

    @abstractmethod
    def decode(self, x: np.array):
        """
        Transform features from encoded to original representation.
        
        This method reverses the encoding transformation, converting features from
        the encoded space back to their original representation. This is essential
        for making explanations interpretable to users.
        
        Args:
            x (np.array): Array of shape (n_samples, n_encoded_features) containing
                samples in the encoded feature space
        
        Returns:
            np.array: Decoded array of shape (n_samples, n_features) in the original
                feature space
        
        Example:
            >>> # Encoded:  [[1, 0, 0, 25], [0, 1, 0, 30]]
            >>> # Original: [['red', 25], ['blue', 30]]
            >>> decoded = encoder.decode(encoded_data)
        """
        return


    @abstractmethod
    def decode_target_class(self, x: np.array):
        return

    @abstractmethod
    def encode_target_class(self, param):
        pass