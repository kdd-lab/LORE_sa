import itertools

from sklearn.compose import ColumnTransformer

from .enc_dec import EncDec
import numpy as np
import copy

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, OrdinalEncoder

__all__ = ["EncDec", "ColumnTransformerEnc"]


class InvertableColumnTransformer(ColumnTransformer):
    """
    Adds an inverse transform method to the standard sklearn.compose.ColumnTransformer.

    Warning this is flaky and use at your own risk.  Validation checks that the column count in
    `transformers` are in your object `X` to be inverted.  Reordering of columns will break things!

    taken from: https://github.com/scikit-learn/scikit-learn/issues/11463
    """
    def inverse_transform(self, X:np.array):
        # print(X)
        arrays = []
        for name, indices in self.output_indices_.items():
            transformer = self.named_transformers_.get(name, None)
            # print(name, indices.start, indices.stop, transformer)
            arr = X[:, indices.start: indices.stop]
            if transformer in (None, "passthrough", "drop"):
                pass
            elif arr.size >0:
                arr = transformer.inverse_transform(arr)
            arrays.append(arr)
        retarr = np.concatenate(arrays, axis=1)
        reordered_array = np.zeros_like(retarr)

        # apply the original order of the columns
        dest_indexes = []
        for t in self.transformers_:
            dest_indexes.extend(t[2])

        for i, d in enumerate(dest_indexes):
            reordered_array[:, d] = retarr[:, i]

        return reordered_array


class ColumnTransformerEnc(EncDec):
    """
    It provides an interface to access One Hot enconding (https://en.wikipedia.org/wiki/One-hot) functions.
    It relies on OneHotEncoder class from sklearn
    """

    def __init__(self,descriptor: dict):
        super().__init__(descriptor)
        self.type='one-hot'
        self.encoded_descriptor = copy.deepcopy(self.dataset_descriptor)
        self.intervals = None # intervals of indexes to map the features to the one-hot encoded features

        # from the dataset descriptor, we extract the number of features, calculating the maximum
        # index of the features in the sub-fields of the descriptor: categorical, numeric, target
        max_index = 0
        for l in ['numeric', 'categorical', 'ordinal', 'target']:
            if l in self.dataset_descriptor:
                for k, v in self.dataset_descriptor[l].items():
                    if v['index'] > max_index:
                        max_index = v['index']
        categories = np.empty(max_index+1, dtype=object)

        for l in ['numeric', 'categorical', 'ordinal', 'target']: #remove target from the scan
            if l in self.dataset_descriptor:
                if l == 'numeric':
                    for k, v in self.dataset_descriptor[l].items():
                        categories[v['index']] = [v['min'], v['max']]
                else:
                    for k, v in self.dataset_descriptor[l].items():
                        categories[v['index']] = v['distinct_values']

        # create a dataset to fit the encoder with the categories
        # each entry of the datasets should contain the corersponding values in the right position
        # of the categories array


        # compute the max length of the categories
        max_len = max([ len(c) for c in categories])
        for i, c in enumerate(categories):
            if len(c) < max_len:
                repetitions = max_len // len(c) + 1
                categories[i] = np.tile(c, repetitions)[:max_len]

        mock_data = list(map(list, zip(*categories)))
        # extract the column index of the target attribute from descriptor
        target_index = -1
        for k, v in self.dataset_descriptor['target'].items():
            target_index = v['index']
        # from mock_data, separate the target column in a separate array
        target_column = [row.pop(target_index) for row in mock_data]


        # Create the column transformer to apply OneHotEncoder only to categorical features
        self.encoder = InvertableColumnTransformer(
            transformers=[
                ('numeric', FunctionTransformer(lambda x: x), [ v['index'] for v in self.dataset_descriptor['numeric'].values()]),
                ('categorical',
                     OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.int16),
                     [ v['index'] for v in self.dataset_descriptor['categorical'].values()]
                ),
                ('ordinal',
                     OrdinalEncoder(dtype=np.int16),
                     [v['index'] for v in self.dataset_descriptor['ordinal'].values()]
                )
            ],
            remainder='passthrough'
        )
        self.target_encoder = OrdinalEncoder(dtype=np.int16)

        self.encoder.fit(mock_data)
        self.target_encoder.fit(np.array(target_column).reshape(-1, 1))

        # print('transformers', self.encoder.transformers_)
        # print('output indices', self.encoder.output_indices_)
        # print('named transformers', self.encoder.named_transformers_.get('categorical').categories_)


        encoded_features = {}
        for name, indices in self.encoder.output_indices_.items():
            # print(name, indices.start, indices.stop)
            if (indices.start != indices.stop): # make sure the transformer is not a passthrough
                if (name == 'categorical'):
                    # print(self.encoder.named_transformers_.get(name).categories_)
                    cat_categories = self.encoder.named_transformers_.get(name).categories_
                    i = indices.start
                    for j, k in enumerate(self.dataset_descriptor['categorical'].keys()):
                        # print('categorical', k,  cat_categories[j], self.dataset_descriptor['categorical'][k]['index'], i)
                        self.encoded_descriptor['categorical'][k]['index'] = i
                        for v in cat_categories[j]:
                            # print('   ', i, f"{k}={v}")
                            encoded_features[i] = f"{k}={v}"
                            i += 1
                if (name == 'ordinal'):
                    # print(self.encoder.named_transformers_.get(name).categories_)
                    cat_categories = self.encoder.named_transformers_.get(name).categories_
                    i = indices.start
                    for j, k in enumerate(self.dataset_descriptor['ordinal'].keys()):
                        # print('categorical', k,  cat_categories[j], self.dataset_descriptor['categorical'][k]['index'], i)
                        self.encoded_descriptor['ordinal'][k]['index'] = i
                        for v in cat_categories[j]:
                            # print('   ', i, f"{k}={v}")
                            encoded_features[i] = f"{k}={v}"
                            i += 1
                if name == 'target':
                    # print(self.encoder.named_transformers_.get(name).categories_)
                    target_categories = self.encoder.named_transformers_.get(name).categories_
                    i = indices.start
                    for j, k in enumerate(self.dataset_descriptor['target'].keys()):
                        # print('target', k,  target_categories[j], self.dataset_descriptor['target'][k]['index'], i)
                        self.encoded_descriptor['target'][k]['index'] = i
                        encoded_features[i] = k
                        # for v in target_categories[j]:
                            # print('   ', i, f"{k}={v} [{self.encoder.named_transformers_.get(name).transform([[v]])[0]}]")
                        i += 1
                if name == 'numeric':
                    # print('numeric', indices.start, indices.stop)
                    i = indices.start
                    for k, v in self.dataset_descriptor['numeric'].items():
                        # print('   ',i , k)
                        self.encoded_descriptor['numeric'][k]['index'] = i
                        encoded_features[i] = k
                        i += 1
        # print('encoded features', encoded_features)
        self.encoded_features = encoded_features

        if self.dataset_descriptor.get("categorical") is None:
            raise Exception("Dataset descriptor is malformed for One-Hot Encoder: 'categorical' key is not present")

    def encode(self, X: np.array):
        """
        It applies the encoder to the input features

        :param [Numpy array] x: Array to encode
        :return [Numpy array]: Encoded array
        """
        encoded = self.encoder.transform(X)

        return encoded

    def get_encoded_features(self):
        return dict(sorted(self.encoded_features.items()))

    def get_encoded_intervals(self):
        if self.intervals is None:
            enc_features = self.get_encoded_features()
            start = 0
            end = 1
            self.intervals = []
            for j in range(1, len(enc_features)):
                f = enc_features[j]
                prev_prefix = enc_features[j-1].split('=')[0]
                curr_prefix = f.split('=')[0]
                if curr_prefix != prev_prefix:
                    self.intervals.append([start, end])
                    start = end
                end += 1
            self.intervals.append([start, end])

        return self.intervals

    def __str__(self):
        if len(self.encoded_features) > 0:
            return "ColumnTransformerEncoder - features encoded: %s" % (",".join(self.encoded_features.values()))
        else:
            return "ColumnTransformerEncoder - no features encoded"

    def decode(self, Z: np.array):
        """
        Decode the array staring from the original descriptor

        :param [Numpy array] x: Array to decode
        :return [Numpy array]: Decoded array
        """
        decoded = self.encoder.inverse_transform(Z)
        # print('decoded inverted transformer', decoded)
        # print('encoded feature scikit', self.encoder.named_transformers_.get('categorical').categories_)

        return decoded

    def decode_target_class(self, Z: np.array):
        """
        Decode the target class

        :param [Numpy array] x: Array containing the target class values to be decoded
        """

        return self.target_encoder.inverse_transform(Z)

    def encode_target_class(self, X: np.array):
        """
        Encode the target class
        :param X:
        :return:
        """
        return self.target_encoder.transform(X)
