import os
import unittest

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.neighgen import RandomGenerator
from lore_sa.surrogate import DecisionTreeSurrogate

class SurrogateTest(unittest.TestCase):

    def setUp(self):
        self.dataset = TabularDataset.from_csv('resources/adult.csv', class_name='class')
        self.dataset.df.dropna(inplace=True)
        self.dataset.df.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)
        self.dataset.update_descriptor()
        # print('descriptor', self.dataset.descriptor)
        self.enc = ColumnTransformerEnc(self.dataset.descriptor)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), [0,8,9,10]),
                ('cat', OrdinalEncoder(), [1,2,3,4,5,6,7,11])
            ]
        )

        model_pkl_file = "resources/adult_random_forest.pkl"
        if os.path.exists(model_pkl_file):
            model = joblib.load(model_pkl_file)
        else:
            model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))

            X_train, X_test, y_train, y_test = train_test_split(self.dataset.df.loc[:, 'age':'native-country'].values, self.dataset.df['class'].values,
                                                                test_size=0.3, random_state=42, stratify=self.dataset.df['class'].values)



            model.fit(X_train, y_train)
            joblib.dump(model, model_pkl_file)

        self.bbox = sklearn_classifier_bbox.sklearnBBox(model)

    def test_init(self):
        iris = load_iris()
        dt = DecisionTreeSurrogate().train(iris.data, iris.target)
        self.assertIsInstance(dt, DecisionTreeClassifier)


    def test_fit_from_random_neighborhood(self):
        num_row = 10
        x = self.dataset.df.iloc[num_row][:-1] # remove the class feature from the input instance
        z = self.enc.encode([x.values])[0]

        gen = RandomGenerator(bbox=self.bbox, dataset=self.dataset, encoder=self.enc, ocr=0.1)
        neighbour = gen.generate(z, 1000, self.dataset.descriptor, self.enc)

        # split neighbor in features and class using train_test_split
        neighb_train_Z = neighbour[:, :]
        neighb_train_X = self.enc.decode(neighb_train_Z)
        neighb_train_y = self.bbox.predict(neighb_train_X)
        neighb_train_yz = self.enc.encode_target_class(neighb_train_y.reshape(-1, 1)).squeeze()

        dt = DecisionTreeSurrogate()
        dt.train(neighb_train_Z, neighb_train_yz)
        # print(export_text(dt.dt, feature_names=list(self.enc.encoded_features.values())[:-1]))
        rule = dt.get_rule(z, self.enc)
        print('rule', rule)
        crules, deltas = dt.get_counterfactual_rules(z, neighb_train_Z, neighb_train_yz, self.enc)
        print('\n crules')
        for c in crules:
            print(c)
        for d in deltas:
            print(d)

if __name__ == '__main__':
    unittest.main()
