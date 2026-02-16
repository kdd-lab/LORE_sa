import unittest
import pandas

from lore_sa.dataset import TabularDataset

descriptor_dummy = {'categorical': {'col3': {'count': {'America': 1, 'Europe': 1},
                        'distinct_values': ['America', 'Europe'],
                        'index': 2}},
                     'numeric': {'col1': {'index': 0,
                                          'max': 2,
                                          'mean': 1.5,
                                          'median': 1.5,
                                          'min': 1,
                                          'q1': 1.25,
                                          'q3': 1.75,
                                          'std': 0.7071067811865476},
                                 'col2': {'index': 1,
                                          'max': 4,
                                          'mean': 3.5,
                                          'median': 3.5,
                                          'min': 3,
                                          'q1': 3.25,
                                          'q3': 3.75,
                                          'std': 0.7071067811865476}},
                    'ordinal':{}}

class DatasetTest(unittest.TestCase):


    def test_dataset_class_init_from_csv(self):
        # when
        dataset = TabularDataset.from_csv("resources/adult.csv")
        # then
        self.assertIs(dataset.filename,"resources/adult.csv")
        self.assertIsInstance(dataset.df, pandas.DataFrame)

    def test_dataset_class_init_from_dict(self):
        dataset = TabularDataset.from_dict({'col1': [1, 2], 'col2': [3, 4]})

        self.assertIsInstance(dataset.df, pandas.DataFrame)

    def test_dataset_set_class_name(self):
        name = "workclass"
        dataset = TabularDataset.from_csv("resources/adult.csv")
        dataset.set_class_name(name)
        self.assertEqual(dataset.class_name,name)

    def test_dataset_set_class_name_error_list(self):
        name = ["workclass","race"]
        dataset = TabularDataset.from_csv("resources/adult.csv")
        dataset.set_class_name(name)
        self.assertEqual(dataset.class_name,name)

    def test_dataset_set_class_name_initialization(self):
        name = "workclass"
        dataset = TabularDataset.from_csv("resources/adult.csv", class_name = name)
        self.assertEqual(dataset.class_name, name)

    def test_dataset_get_original_dataset(self):
        dataset = TabularDataset.from_dict({'col1': [1, 2], 'col2': [3, 4]})
        dataset.df.drop(columns=["col1"],inplace=True)

        self.assertEqual(len(dataset.df.columns), 1)

    def test_get_numeric_columns(self):
        dataset = TabularDataset.from_dict({'col1': [1, 2], 'col2': [3, 4], 'col3': ['America','Europe']})
        self.assertEqual(dataset.get_numeric_columns(),['col1','col2'])

    def test_get_class_value_as_exptected(self):
        dataset = TabularDataset.from_dict({'col1': [1, 2], 'col2': [3, 4], 'col3': ['America', 'Europe']},class_name='col3')
        self.assertEqual(dataset.get_class_values().tolist(),['America', 'Europe'])

    def test_get_class_value_raise_error(self):
        dataset = TabularDataset.from_dict({'col1': [1, 2], 'col2': [3, 4], 'col3': ['America', 'Europe']})
        with self.assertRaises(Exception) as context:
            dataset.get_class_values()
            self.assertEqual("ERR: class_name is None. Set class_name with set_class_name('<column name>')", str(context.exception))

    def test_update_descriptor(self):
        dataset = TabularDataset.from_dict({'col1': [1, 2], 'col2': [3, 4], 'col3': ['America', 'Europe']})
        descriptor = dataset.descriptor
        self.assertEqual(descriptor,descriptor_dummy)

    def test_get_features_name_by_index(self):
        dataset = TabularDataset.from_dict({'col1': [1, 2], 'col2': [3, 4], 'col3': ['America', 'Europe']})

        self.assertEqual(dataset.get_feature_name(0),'col1')
        self.assertEqual(dataset.get_feature_name(1),'col2')
        self.assertEqual(dataset.get_feature_name(2),'col3')

    def test_get_features_name_by_index_with_class_name(self):
        dataset = TabularDataset.from_dict({'col1': [1, 2], 'col2': [3, 4], 'col3': ['America', 'Europe']},class_name="col3")

        descriptor = dataset.descriptor
        self.assertEqual(descriptor['target'],{'col3': {'count': {'America': 1, 'Europe': 1},
                        'distinct_values': ['America', 'Europe'],
                        'index': 2}})


if __name__ == '__main__':
    unittest.main()