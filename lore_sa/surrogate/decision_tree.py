import copy
import datetime
import operator
from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix

from ..encoder_decoder import EncDec, ColumnTransformerEnc
from ..logger import logger
from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection
from sklearn.experimental import enable_halving_search_cv

__all__ = ["Surrogate", "DecisionTreeSurrogate"]

from ..rule import Expression, Rule
from .surrogate import Surrogate
from ..util import vector2dict, multilabel2str


class DecisionTreeSurrogate(Surrogate):

    def __init__(self, kind=None, preprocessing=None, class_values=None, multi_label: bool = False,
                 one_vs_rest: bool = False, cv=5, prune_tree: bool = False, ):
        super().__init__(kind, preprocessing)
        self.dt = None
        self.fidelity = None
        self.confusion_matrix = None
        self.prune_tree = prune_tree
        self.class_values = class_values
        self.multi_label = multi_label
        self.one_vs_rest = one_vs_rest
        self.cv = cv


    def train(self, Z, Yb, weights=None, ):
        """

        :param Z: The training input samples
        :param Yb: The target values (class labels) as integers or strings.
        :param weights: Sample weights.
        :param class_values:
        :param [bool] multi_label:
        :param [bool] one_vs_rest:
        :param [int] cv:
        :param [bool] prune_tree:
        :return:
        """
        self.dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
        if self.prune_tree is True:
            param_list = {'min_samples_split': [0.01, 0.05, 0.1, 0.2, 3, 2],
                          'min_samples_leaf': [0.001, 0.01, 0.05, 0.1, 2, 4],
                          'splitter': ['best', 'random'],
                          'max_depth': [None, 2, 10, 12, 16, 20, 30],
                          'criterion': ['entropy', 'gini'],
                          'max_features': [0.2, 1, 5, 'sqrt', 'log2']
                          }

            # if multi_label is False or (multi_label is True and one_vs_rest is True):
            #     if len(class_values) == 2 or (multi_label and one_vs_rest):
            #         scoring = 'precision'
            #     else:
            #         scoring = 'precision_macro'
            # else:
            scoring = 'precision_micro'

            dt_search = sklearn.model_selection.HalvingGridSearchCV(self.dt, param_grid=param_list, scoring=scoring,
                                                                    cv=self.cv, n_jobs=-1)
            logger.info('Search the best estimator')
            logger.info('Start time: {0}'.format(datetime.datetime.now()))
            dt_search.fit(Z, Yb, sample_weight=weights)
            logger.info('End time: {0}'.format(datetime.datetime.now()))
            self.dt = dt_search.best_estimator_
            logger.info('Pruning')
            self.prune_duplicate_leaves(self.dt)
        else:
            self.dt.fit(Z, Yb)

        self.fidelity = self.dt.score(Z, Yb)
        self.confusion_matrix = confusion_matrix(Yb, self.dt.predict(Z))
        logger.info("Fidelity: {}".format(self.fidelity))

        return self.dt

    def is_leaf(self, inner_tree, index):
        """Check whether node is leaf node"""
        return (inner_tree.children_left[index] == TREE_LEAF and
                inner_tree.children_right[index] == TREE_LEAF)

    def prune_index(self, inner_tree, decisions, index=0):
        """
        Start pruning from the bottom - if we start from the top, we might miss
        nodes that become leaves during pruning.
        Do not use this directly - use prune_duplicate_leaves instead.
        """
        if not self.is_leaf(inner_tree, inner_tree.children_left[index]):
            self.prune_index(inner_tree, decisions, inner_tree.children_left[index])
        if not self.is_leaf(inner_tree, inner_tree.children_right[index]):
            self.prune_index(inner_tree, decisions, inner_tree.children_right[index])

        # Prune children if both children are leaves now and make the same decision:
        if (self.is_leaf(inner_tree, inner_tree.children_left[index]) and
                self.is_leaf(inner_tree, inner_tree.children_right[index]) and
                (decisions[index] == decisions[inner_tree.children_left[index]]) and
                (decisions[index] == decisions[inner_tree.children_right[index]])):
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
            logger.info("Pruned {}".format(index))

    def prune_duplicate_leaves(self, dt):
        """Remove leaves if both"""
        decisions = dt.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
        self.prune_index(dt.tree_, decisions)

    def get_rule(self, z: np.array, encoder: EncDec = None):
        """
        Extract the rules as the promises and consequences {p -> y}, starting from a Decision Tree

             {( income > 90) -> grant),
                ( job = employer) -> grant)
            }

        :param [Numpy Array] z: instance encoded of the dataset to extract the rule
        :param [EncDec] encdec:
        :return [Rule]: Rule objects
        """
        z = z.reshape(1, -1)
        feature = self.dt.tree_.feature
        threshold = self.dt.tree_.threshold
        predicted_class = self.dt.predict(z)
        inv_transform_predicted_class = encoder.decode_target_class([predicted_class])[0]

        if encoder.type == 'one_hot':
            target_feature_name = list(encoder.encoded_descriptor['target'].keys())[0]
        else:
            target_feature_name = 'target'

        consequence = Expression(variable=target_feature_name, operator=operator.eq,
                                 value=inv_transform_predicted_class[0])

        leave_id = self.dt.apply(z)
        node_index = self.dt.decision_path(z).indices

        if encoder.type == 'one-hot':
            feature_names = list(encoder.encoded_features.values())
            numeric_columns = list(encoder.encoded_descriptor['numeric'].keys())
        else:
            feature_names = [f"dim {i}" for i in range(self.dt.tree_.n_features)]
            numeric_columns = feature_names

        premises = list()
        for node_id in node_index:
            if leave_id[0] == node_id:
                break
            else:
                attribute = feature_names[feature[node_id]]
                if attribute not in numeric_columns:
                    # this is a categorical feature
                    # print(f"{attribute} has value {x[0][feature[node_id]]} and threshold is {threshold[node_id]}")
                    thr = False if z[0][feature[node_id]] <= threshold[node_id] else True
                    op = operator.eq
                else:
                    thr = threshold[node_id]
                    op = operator.le if z[0][feature[node_id]] <= threshold[node_id] else operator.gt

                premises.append(Expression(attribute, op, thr))

        premises = self.compact_premises(premises)
        return Rule(premises=premises, consequences=consequence)

    def compact_premises(self, premises_list):
        """
        Remove the same premises with different values of threashold

        :param premises_list: List of Expressions that defines the premises
        :return:
        """
        attribute_list = defaultdict(list)
        for premise in premises_list:
            attribute_list[premise.variable].append(premise)

        compact_plist = list()
        for att, alist in attribute_list.items():
            if len(alist) > 1:
                min_thr = None
                max_thr = None
                for av in alist:
                    if av.operator == operator.le:
                        max_thr = min(av.value, max_thr) if max_thr else av.value
                    elif av.operator == operator.gt:
                        min_thr = max(av.value, min_thr) if min_thr else av.value

                if max_thr:
                    compact_plist.append(Expression(att, operator.le, max_thr))

                if min_thr:
                    compact_plist.append(Expression(att, operator.gt, min_thr))
            else:
                compact_plist.append(alist[0])
        return compact_plist

    def get_counterfactual_rules(self, z: np.array, neighborhood_train_X: np.array, neighborhood_train_Y: np.array,
                                 encoder: EncDec = None,
                                 filter_crules=None, constraints: dict = None, unadmittible_features: list = None):


        feature_names = list(encoder.encoded_features.values())
        # numeric_columns = list(encoder.encoded_descriptor['numeric'].keys())
        predicted_class = self.dt.predict(z.reshape(1, -1))[0]
        # inv_transform_predicted_class = encoder.encoder.named_transformers_.get('target')\
        #     .inverse_transform([predicted_class])[0] #TODO: modify to generalize to multiclasses

        if encoder.type == 'one_hot':
            class_name = list(encoder.encoded_descriptor['target'].keys())[0]
            class_values = list(encoder.encoded_descriptor['target'].values())[0]['distinct_values']
        else:
            class_name = 'target'
            class_values = self.dt.tree_.n_classes

        clen = np.inf
        crule_list = list()
        delta_list = list()

        # y = self.dt.predict(neighborhood_dataset.df)[0]
        # Y = self.dt.predict(neighborhood_dataset.df)

        x_dict = vector2dict(z, feature_names)
        # select the subset of ```neighborhood_train_X``` that have a classification different from the input x
        Z1 = neighborhood_train_X[np.where(neighborhood_train_Y != predicted_class)]

        # We search for the shortest rule among those that support the elements in Z1
        for zi in Z1:
            #
            crule = self.get_rule(z=zi, encoder=encoder)

            delta = self.get_falsified_conditions(x_dict, crule)
            num_falsified_conditions = len(delta)

            if unadmittible_features is not None:
                is_feasible = self.check_feasibility_of_falsified_conditions(delta, unadmittible_features)
                if is_feasible is False:
                    continue

            if constraints is not None:
                ##TODO
                to_remove = list()
                for p in crule.premises:
                    if p.variable in constraints.keys():
                        if p.operator == constraints[p.variable]['op']:
                            if p.thr > constraints[p.variable]['thr']:
                                break
                                # caso corretto
                ##TODO

            if filter_crules is not None:
                xc = self.apply_counterfactual(z, delta, feature_names)
                bb_outcomec = filter_crules(xc.reshape(1, -1))[0]
                bb_outcomec = class_values[bb_outcomec] if isinstance(class_name, str) else multilabel2str(bb_outcomec,
                                                                                                           class_values)
                dt_outcomec = crule.cons

                if bb_outcomec == dt_outcomec:
                    if num_falsified_conditions < clen:
                        clen = num_falsified_conditions
                        crule_list = [crule]
                        delta_list = [delta]
                    elif num_falsified_conditions == clen:
                        if delta not in delta_list:
                            crule_list.append(crule)
                            delta_list.append(delta)
            else:
                if num_falsified_conditions < clen and num_falsified_conditions > 0:
                    clen = num_falsified_conditions
                    crule_list = [crule]
                    delta_list = [delta]
                    # print(crule, delta)
                elif num_falsified_conditions == clen:
                    if delta not in delta_list:
                        crule_list.append(crule)
                        delta_list.append(delta)

        return crule_list, delta_list

    def get_falsified_conditions(self, x_dict: dict, crule: Rule):
        """
        Check the wrong conditions.
        Separates logic for numerical inequalities vs boolean/categorical equalities.

        :param x_dict: Dictionary containing instance data.
                       For inequalities: {variable: value}
                       For equalities: {f"{variable} == {value}": boolean_flag}
        :param crule: The rule object containing premises.
        :return: list of falsified premises
        """
        delta = []

        for p in crule.premises:
            # Group 1: Numerical Inequalities (le, lt, ge, gt)
            if p.operator in [operator.le, operator.lt, operator.ge, operator.gt]:
                # Retrieve the raw value (e.g., 'Age': 25)
                if p.variable not in x_dict:
                    continue # Skip if data is missing, or raise ValueError

                actual_value = x_dict[p.variable]

                # We append to delta if the condition is NOT met
                if p.operator == operator.le and not (actual_value <= p.value):
                    delta.append(p)
                elif p.operator == operator.lt and not (actual_value < p.value):
                    delta.append(p)
                elif p.operator == operator.ge and not (actual_value >= p.value):
                    delta.append(p)
                elif p.operator == operator.gt and not (actual_value > p.value):
                    delta.append(p)

            # Group 2: Boolean / Categorical Logic (eq, ne)
            elif p.operator in [operator.eq, operator.ne]:
                # Construct the special key: e.g., "Color == Red"
                # Note: We convert p.value to string to ensure the format matches
                key = f"{p.variable} == {p.value}"

                # Check if this boolean feature exists in x_dict
                if key in x_dict:
                    # x_dict[key] is likely 1/True (Match) or 0/False (No Match)
                    is_match = x_dict[key]

                    # Case A: Premise is EQUAL (==).
                    # Falsified if x_dict says it is NOT a match (False/0)
                    if p.operator == operator.eq and not is_match:
                        delta.append(p)

                    # Case B: Premise is NOT EQUAL (!=).
                    # Falsified if x_dict says it IS a match (True/1)
                    elif p.operator == operator.ne and is_match:
                        delta.append(p)
                else:
                    # Optional: Handle case where the specific one-hot feature is missing
                    # usually in one-hot encoding, if the key is missing, it implies False,
                    # but explicit checks are safer.
                    pass

        return delta

    def check_feasibility_of_falsified_conditions(self, delta, unadmittible_features: list):
        """
        Check if a falsifield confition is in an unadmittible feature list
        :param delta:
        :param unadmittible_features:
        :return: True or False
        """
        for p in delta:
            if p.variable in unadmittible_features:
                if unadmittible_features[p.variable] is None:
                    return False
                else:
                    if unadmittible_features[p.variable] == p.operator:
                        return False
        return True

    def apply_counterfactual(self, x, delta, feature_names:list, features_map=None, features_map_inv=None, numeric_columns=None):
        x_dict = vector2dict(x, feature_names)
        x_copy_dict = copy.deepcopy(x_dict)
        for p in delta:
            if p.variable in numeric_columns:
                if p.value == int(p.value):
                    gap = 1.0
                else:
                    decimals = list(str(p.value).split('.')[1])
                    for idx, e in enumerate(decimals):
                        if e != '0':
                            break
                    gap = 1 / (10 ** (idx + 1))
                if p.operator == operator.gt:
                    x_copy_dict[p.variable] = p.value + gap
                else:
                    x_copy_dict[p.variable] = p.value
            else:
                fn = p.variable
                if p.operator == operator.gt:
                    if features_map is not None:
                        fi = list(feature_names).index(p.att)
                        fi = features_map_inv[fi]
                        for fv in features_map[fi]:
                            x_copy_dict['%s=%s' % (fn, fv)] = 0.0
                    x_copy_dict[p.att] = 1.0

                else:
                    if features_map is not None:
                        fi = list(feature_names).index(p.att)
                        fi = features_map_inv[fi]
                        for fv in features_map[fi]:
                            x_copy_dict['%s=%s' % (fn, fv)] = 1.0
                    x_copy_dict[p.att] = 0.0

        x_counterfactual = np.zeros(len(x_dict))
        for i, fn in enumerate(feature_names):
            x_counterfactual[i] = x_copy_dict[fn]

        return x_counterfactual