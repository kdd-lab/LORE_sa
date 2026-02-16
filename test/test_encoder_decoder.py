import unittest

import pandas as pd
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import ColumnTransformerEnc
import numpy as np



class EncDecTest(unittest.TestCase):

    descriptor_dummy = {}
    def setUp(self):
        # self.descriptor_dummy___ = {'categorical': {'col3': {'count': {'America': 1, 'Europe': 1, 'Africa': 1},
        #                                            'distinct_values': ['America', 'Europe', 'Africa'],
        #                                            'index': 2},
        #                                   'colours': {
        #                                       'distinct_values': ['White', 'Black', 'Red', 'Blue', 'Green'],
        #                                       'index': 4
        #                                   }},
        #                     'numeric': {'col1': {'index': 0,'max': 2,'mean': 1.5,'median': 1.5,'min': 1,'q1': 1.25,
        #                                          'q3': 1.75,'std': 0.7071067811865476},
        #                                 'col2': {'index': 1,'max': 4,'mean': 3.5,'median': 3.5,'min': 3,'q1': 3.25,
        #                                          'q3': 3.75,'std': 0.7071067811865476}},
        #                     'target': {'education': {'index': 3,
        #                                               'distinct_values': ['Elementary', 'High School', 'College',
        #                                                                   'Graduate', 'Post-graduate']}
        #                                 }}

        self.tabular_descriptor = {'categorical': {'col3': {'count': {'America': 1, 'Europe': 1, 'Africa': 1},
                                                        'distinct_values': ['America', 'Europe', 'Africa'],
                                                        'index': 2},
                                               'colours': {
                                                   'distinct_values': ['White', 'Black', 'Red', 'Blue', 'Green'],
                                                   'index': 3
                                               }},
                                 'numeric': {
                                     'col1': {'index': 0, 'max': 2, 'mean': 1.5, 'median': 1.5, 'min': 1, 'q1': 1.25,
                                              'q3': 1.75, 'std': 0.7071067811865476},
                                     'col2': {'index': 1, 'max': 4, 'mean': 3.5, 'median': 3.5, 'min': 3, 'q1': 3.25,
                                              'q3': 3.75, 'std': 0.7071067811865476}},
                                'ordinal' : {},
                                 'target': {'education': {'index': 4,
                                                           'distinct_values': ['Elementary', 'High School', 'College',
                                                                               'Graduate', 'Post-graduate']}
                                             }}

        # We create an instance from a dataset to check the correct work of the pair encoder-decoder
        adult_df = pd.read_csv('resources/adult.csv',skipinitialspace=True, na_values='?', keep_default_na=True)
        adult_df.dropna(inplace=True)

        self.dataset = TabularDataset(adult_df, class_name='class')
        self.dataset.df.dropna(inplace=True)
        self.dataset.update_descriptor(ordinal_columns=['education'])


    def test_column_transformer_encoder_init(self):
        one_hot_enc = ColumnTransformerEnc(self.tabular_descriptor)
        self.assertEqual(one_hot_enc.type,'one-hot')
        # We always get back a string with the features encoded. So this test can be removed
        # self.assertEqual(one_hot_enc.__str__(),"OneHotEncoder - no features encoded")

    def test_one_hot_init_no_ordinal_field(self):
        wrong_descriptor = {}
        with self.assertRaises(Exception) as context:
            one_hot_enc = ColumnTransformerEnc(wrong_descriptor)
            self.assertEqual("Dataset descriptor is malformed for One-Hot Encoder: 'categorical' key is not present", str(context.exception))

    def test_column_transformer_encoder_init_with_features_encoder(self):
        one_hot_enc = ColumnTransformerEnc(self.tabular_descriptor)
        encoded = one_hot_enc.encode([[1, 2, "Europe", "Green"]])  # "Graduate"
        self.assertEqual("ColumnTransformerEncoder - features encoded: col1,col2,col3=Africa,col3=America,col3=Europe,colours=Black,colours=Blue,colours=Green,colours=Red,colours=White", one_hot_enc.__str__())
        self.assertEqual(encoded.tolist(),np.array([1, 2, 0, 0, 1, 0, 0, 1, 0, 0]).reshape(1, -1).tolist())
        self.assertEqual(one_hot_enc.encoded_descriptor['numeric']['col1']['index'], 0)
        self.assertEqual(one_hot_enc.encoded_descriptor['numeric']['col2']['index'], 1)
        # self.assertEqual(one_hot_enc.encoded_descriptor['target']['education']['index'], 10)
        self.assertEqual(one_hot_enc.get_encoded_features(), {0: 'col1',
                                                                     1: 'col2',
                                                                     2: 'col3=Africa',
                                                                     3: 'col3=America',
                                                                     4: 'col3=Europe',
                                                                     5: 'colours=Black',
                                                                     6: 'colours=Blue',
                                                                     7: 'colours=Green',
                                                                     8: 'colours=Red',
                                                                     9: 'colours=White'})

    def test_column_transformer_decode_init_with_features_encoder(self):
        one_hot_enc = ColumnTransformerEnc(self.tabular_descriptor)
        decoded = one_hot_enc.decode(np.array([1, 2, 1, 0, 0, 0, 1, 0, 0, 0]).reshape(1, -1))
        decoded_target = one_hot_enc.decode_target_class(np.array([ 2]).reshape(1, -1))

        self.assertEqual(decoded.tolist(), [[1, 2, "Africa", "Blue"]])
        self.assertEqual(decoded_target.tolist(), [[ "Graduate"]] )


    def test_tabular_encode(self):
        # Deprecated, since we can use ColumnTransformerEnc
        tabular_enc = ColumnTransformerEnc(self.tabular_descriptor)
        encoded = tabular_enc.encode([[1, 2, "Europe", "Green"]])
        self.assertEqual(encoded.tolist(),np.array([1, 2, 0, 0, 1, 0, 0, 1, 0, 0]).reshape(1, -1).tolist())

    def test_tabular_decode(self):
        tabular_enc = ColumnTransformerEnc(self.tabular_descriptor)
        decoded = tabular_enc.decode(np.array([1, 2, 1, 0, 0, 0, 0, 0, 0, 1, 3]).reshape(1, -1)).squeeze()
        self.assertEqual( [1, 2, "Africa", 'White'], decoded.tolist())


    def test_creation_of_descriptor_from_dataset(self):
        tabular_enc = ColumnTransformerEnc(self.dataset.descriptor)
        ref_value = self.dataset.df.iloc[0].values[:-1]
        encoded = tabular_enc.encode([ref_value])
        decoded = tabular_enc.decode(encoded)
        self.assertTrue(np.array_equal(ref_value, decoded[0]))

if __name__ == '__main__':
    unittest.main()
