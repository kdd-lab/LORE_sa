from abc import ABC, abstractmethod

__all__ = ["AbstractBBox"]

class AbstractBBox(ABC):
    """
    Abstract base class for wrapping black box models.
    
    This class provides a unified interface for different types of black box models
    (e.g., scikit-learn, Keras, PyTorch) to be explained by LORE. Any black box model
    must be wrapped in a subclass of AbstractBBox that implements the predict() and
    predict_proba() methods.
    
    The wrapper allows LORE to interact with any machine learning model in a consistent
    way, regardless of the underlying framework or implementation.
    
    Attributes:
        classifier: The underlying black box model
    
    Methods:
        predict: Predict class labels for samples
        predict_proba: Predict class probabilities for samples
        model: Return the underlying model
    
    Example:
        To create a wrapper for a custom model:
        
        >>> from lore_sa.bbox import AbstractBBox
        >>> 
        >>> class MyCustomBBox(AbstractBBox):
        ...     def __init__(self, model):
        ...         self.classifier = model
        ...     
        ...     def predict(self, X):
        ...         return self.classifier.predict(X)
        ...     
        ...     def predict_proba(self, X):
        ...         return self.classifier.predict_proba(X)
    
    See Also:
        sklearn_classifier_bbox.sklearnBBox: Pre-built wrapper for scikit-learn models
        keras_classifier_wrapper: Pre-built wrapper for Keras models
    """

    def __init__(self, classifier):
        """
        Initialize the black box wrapper.
        
        Args:
            classifier: The machine learning model to wrap. The model should have
                methods compatible with the predict() and predict_proba() interface.
        """
        pass


    def model(self):
        """
        Return the underlying black box model.
        
        Returns:
            The wrapped classifier/model object.
        """
        return self.model()

    @abstractmethod
    def predict(self, sample_matrix: list):
        """
        Predict class labels for samples (sklearn-like interface).
        
        This method wraps the underlying model's predict method to provide a
        consistent interface for LORE to get class predictions.
        
        Args:
            sample_matrix (array-like): Array of shape (n_samples, n_features) 
                containing the samples to predict. Can be a list, numpy array,
                or sparse matrix depending on the model's requirements.
        
        Returns:
            np.ndarray: Array of shape (n_samples,) containing the predicted 
                class labels for each sample.
        
        Example:
            >>> predictions = bbox.predict([[1, 2, 3], [4, 5, 6]])
            >>> print(predictions)  # e.g., array([0, 1])
        """
        pass

    @abstractmethod
    def predict_proba(self, sample_matrix: list):
        """
        Predict class probabilities for samples (sklearn-like interface).
        
        This method wraps the underlying model's predict_proba method to provide
        probability estimates for each class for the given samples.
        
        Args:
            sample_matrix (array-like): Array of shape (n_samples, n_features) 
                containing the samples to predict. Can be a list, numpy array,
                or sparse matrix depending on the model's requirements.
        
        Returns:
            np.ndarray: Array of shape (n_samples, n_classes) containing the 
                class probabilities for each sample. Each row sums to 1.0.
        
        Example:
            >>> probas = bbox.predict_proba([[1, 2, 3], [4, 5, 6]])
            >>> print(probas)  # e.g., array([[0.8, 0.2], [0.3, 0.7]])
        
        Note:
            For multi-class classification, the array has shape (n_samples, n_classes).
            For binary classification, it typically has shape (n_samples, 2).
        """
        pass