import numpy

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, \
    CategoricalHyperparameter

from autosklearn.pipeline.components.base import \
    AutoSklearnClassificationAlgorithm

from autosklearn.pipeline.constants import *


class LightGBMClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(self, learning_rate, n_estimators,
                 num_leaves, max_bin, nthread=1,
                 random_state=None, verbose=0):

        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.max_bin = max_bin

        self.nthread = nthread

        # Whether to print messages while running boosting.
        if verbose:
            self.silent = False
        else:
            self.silent = True

        # Random number seed.
        if random_state is None:
            self.seed = numpy.random.randint(1, 10000, size=1)[0]
        else:
            self.seed = random_state.randint(1, 10000, size=1)[0]

        self.estimator = None

    def fit(self, X, y):
        import lightgbm as lgb

        self.learning_rate = float(self.learning_rate)
        self.n_estimators = int(self.n_estimators)

        self.num_leaves = int(self.num_leaves)
        self.max_bin = int(self.max_bin)

        self.estimator = lgb.LGBMClassifier(
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                max_bin=self.max_bin,
                silent=self.silent,
                nthread=self.nthread,
                seed=self.seed
                )
        self.estimator.fit(X, y, verbose=self.filent)

        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LGB',
                'name': 'LightGBM  Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        # Parameterized Hyperparameters
        num_leaves = UniformIntegerHyperparameter(
            name="num_leaves", lower=3, upper=1023, default_value=31)
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default_value=100)
        max_bin = UniformIntegerHyperparameter(
            name="max_bin", lower=5, upper=255, default_value=255)

        cs.add_hyperparameters([num_leaves, learning_rate, n_estimators, max_bin])
        return cs
