from lore_sa.bbox.bbox import AbstractBBox

__all__ = ["AbstractBBox","sklearn_ts_classifier_wrapper"]
class sklearn_ts_classifier_wrapper(AbstractBBox):

    def __init__(self, classifier):
        super().__init__()
        self.bbox = classifier


    def predict(self, X):
        #change the input shape of the time series, from 3 dimensions to 2.
        X = X[:, :, 0]
        return self.bbox.predict(X).ravel()

    def predict_proba(self, X):
        # change the input shape of the time series, from 3 dimensions to 2.
        X = X[:, :, 0]
        return self.bbox.predict_proba(X)