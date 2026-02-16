import os
import unittest

import joblib
import sklearn.datasets
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from lore_sa.bbox import AbstractBBox, sklearn_classifier_bbox, bbox
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.lore import TabularRandomGeneratorLore, TabularGeneticGeneratorLore


class LoremTest(unittest.TestCase):

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

        self.tabularLore = TabularGeneticGeneratorLore(self.bbox, self.dataset)

    def test_lorem_init(self):
        # given

        # when

        # then
        pass

    def test_explain_instance(self):
        # given
        num_row = 10
        x = self.dataset.df.iloc[num_row][:-1]
        # when
        explanation = self.tabularLore.explain(x)
        # then
        print(explanation)
        pass



if __name__ == '__main__':
    unittest.main()
