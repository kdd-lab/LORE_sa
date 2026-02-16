import os
import joblib
import unittest

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.neighgen import RandomGenerator
from lore_sa.neighgen.genetic import GeneticGenerator, LegacyGeneticGenerator
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator


class NeighgenTest(unittest.TestCase):

    def setUp(self):
        self.dataset = TabularDataset.from_csv('resources/adult.csv', class_name='class')
        self.dataset.df.dropna(inplace=True)
        self.dataset.df.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)
        self.dataset.update_descriptor()
        # print('descriptor', self.dataset.descriptor)
        self.enc = ColumnTransformerEnc(self.dataset.descriptor)

        model_pkl_file = "resources/adult_random_forest.pkl"
        if os.path.exists(model_pkl_file):
            model = joblib.load(model_pkl_file)
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), [0,8,9,10]),
                    ('cat', OrdinalEncoder(), [1,2,3,4,5,6,7,11])
                ]
            )
            model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))

            X_train, X_test, y_train, y_test = train_test_split(self.dataset.df.loc[:, 'age':'native-country'].values, self.dataset.df['class'].values,
                                                                test_size=0.3, random_state=42, stratify=self.dataset.df['class'].values)



            model.fit(X_train, y_train)
            joblib.dump(model, model_pkl_file)

        self.bbox = sklearn_classifier_bbox.sklearnBBox(model)

    def test_random_generator(self):
        gen = RandomGenerator(bbox=self.bbox, dataset=self.dataset, encoder=self.enc, ocr=0.1)
        self.assertIsInstance(gen, NeighborhoodGenerator)
        self.assertIsInstance(gen, RandomGenerator)

    def test_random_generator_generate_balanced(self):
        num_row = 10
        x = self.dataset.df.iloc[num_row][:-1] # remove the class feature from the input instance
        z = self.enc.encode([x.values])[0]

        gen = RandomGenerator(bbox=self.bbox, dataset=self.dataset, encoder=self.enc, ocr=0.1)
        neighbour = gen.generate(z, 1000, self.dataset.descriptor, self.enc)
        # Assert the lenght of the generated dataset is at least 1000
        self.assertGreaterEqual(neighbour.shape[0], 1000)

    def test_random_generator_generate_raw(self):
        num_row = 10
        x = self.dataset.df.iloc[num_row][:-1] # remove the class feature from the input instance
        z = self.enc.encode([x.values])[0]

        gen = RandomGenerator(bbox=self.bbox, dataset=self.dataset, encoder=self.enc, ocr=0.1)
        neighbour = gen.generate(z, 1000, self.dataset.descriptor, self.enc)
        self.assertGreaterEqual(neighbour.shape[0], 1000)
        self.assertEqual(neighbour.shape[1], len(z))


    def test_genetic_generator(self):
        gen = GeneticGenerator()
        self.assertIsInstance(gen, NeighborhoodGenerator)
        self.assertIsInstance(gen, GeneticGenerator)

    def test_genetic_generator_generate_balanced(self):
        num_row = 100
        x = self.dataset.df.iloc[num_row][:-1]
        z = self.enc.encode([x.values])[0] # remove the class feature from the input instance

        gen = GeneticGenerator(bbox=self.bbox, dataset=self.dataset, encoder=self.enc, ocr=0.1, ngen=20)
        neighbour = gen.generate(z, 1000, self.dataset.descriptor, self.enc)
        # Assert the lenght of the generated dataset is at least 1000
        self.assertGreaterEqual(neighbour.shape[0], 100)

        dec_neighbour = self.enc.decode(neighbour)
        # checking and filtering the rows in dec_neighbour that contains a None value
        # if there is a None value, the row is removed
        dec_neighbour = dec_neighbour[~pd.isnull(dec_neighbour).any(axis=1)]
        pred_neighbour = self.bbox.predict(dec_neighbour)
        classes, count = np.unique(pred_neighbour, return_counts=True)
        print('classes', classes)
        print('count', count)
        self.assertTrue(len(classes) > 1, "The generated dataset should contain at least two classes")

    def test_legacy_genetic_generator_generate_balanced(self):
        num_row = 100
        x = self.dataset.df.iloc[num_row][:-1]
        z = self.enc.encode([x.values])[0] # remove the class feature from the input instance
        gen = LegacyGeneticGenerator(bbox=self.bbox, dataset=self.dataset, encoder=self.enc, ocr=0.1, ngen=20)
        neighbour = gen.generate(z, 100, self.dataset.descriptor, self.enc)
        # Assert the lenght of the generated dataset is at least 1000
        self.assertGreaterEqual(neighbour.shape[0], 100)
        dec_neighbour = self.enc.decode(neighbour)
        # checking and filtering the rows in dec_neighbour that contains a None value
        # if there is a None value, the row is removed
        dec_neighbour = dec_neighbour[~pd.isnull(dec_neighbour).any(axis=1)]
        pred_neighbour = self.bbox.predict(dec_neighbour)
        classes, count = np.unique(pred_neighbour, return_counts=True)
        print('classes', classes)
        print('count', count)
        self.assertTrue(len(classes) > 1, "The generated dataset should contain at least two classes")

if __name__ == '__main__':
    unittest.main()