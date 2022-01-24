from abc import abstractmethod
from copy import deepcopy
from functools import partial
from hyperopt import fmin, tpe, Trials, space_eval
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, List, Tuple
from .architecture import Architecture
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface
from .metric import AUPRC, LogLoss, F1_Macro, AUROC, BrierScore


class Tuner(ConfigurableComponent):

    def __init__(self, config: Dict):
        super().__init__(config)

    @abstractmethod
    def fit(self, architecture: Architecture, x, y) -> Tuple[Architecture, Dict]:
        pass


class RandomSearchTuner(Tuner):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.hyperparameters = config['hyperparameters']
        self.num_iters = config['num_iters']
        self.num_cv_folds = config['num_cv_folds']
        self.scoring_function = config['scoring_function']
        self.verbose = config.get('verbose', False)

    def fit(self, architecture, x, y) -> Tuple[Architecture, Dict]:
        """
        Find optimal hyperparameters for the training of the `Architecture`, yielding a trained `Predictor` object and a
         `training_metadata` dictionary.

        Args:
            architecture: The `Architecture` object to train.
            x: The training set.
            y: The labels for the training set.

        Returns: a trained `Predictor` object and a `training_metadata` dictionary.

        """
        random_search = RandomizedSearchCV(estimator=architecture, param_distributions=self.hyperparameters,
                                           n_iter=self.num_iters, cv=self.num_cv_folds, verbose=self.verbose,
                                           random_state=42, n_jobs=-1, scoring=self.scoring_function)
        random_search.fit(x, y)
        best_est = random_search.best_estimator_
        training_metadata = random_search.best_params_
        return best_est, training_metadata


class BayesianTuner(Tuner):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.hyperparameters = config['hyperparameters']
        self.num_iters = config['num_iters']
        self.num_cv_folds = config['num_cv_folds']
        self.scoring_function = config['scoring_function']
        self.verbose = config['verbose']
        self.timeout = config['hyperopt_timeout']

    def fit(self, architecture, x, y) -> Tuple[Architecture, Dict]:
        cv_ids = list(range(self.num_cv_folds)) * np.floor((len(x) / self.num_cv_folds)).astype(int)
        cv_ids.extend(list(range(len(x) % self.num_cv_folds)))
        cv_ids = np.random.permutation(cv_ids)

        trials = Trials()
        best_fit = fmin(partial(self.hyperopt_objective, model=architecture, x_train=x, y_train=y,
                                scoring_fn=self.scoring_function, ids=cv_ids, nfolds=self.num_cv_folds,
                                verbose=self.verbose),
                        self.hyperparameters, algo=tpe.suggest, timeout=self.timeout, max_evals=self.num_iters,
                        trials=trials, verbose=self.verbose)
        best_params = space_eval(self.hyperparameters, best_fit)
        tuned_architecture = architecture.set_params(**best_params)
        best_est = tuned_architecture.fit(x, y)

        training_metadata = {
            'trials': deepcopy(trials),
            'params': best_params,
        }
        return best_est, training_metadata

    @classmethod
    def hyperopt_objective(cls, params, model, x_train, y_train, scoring_fn: str, ids: List[int], nfolds, verbose):
        """
        Objective to minimise. For use with the hyperopt package, which performs Bayesian hyperparameter searches.
        This takes in the model, data, and a list of parameter values that should be used for calculating the loss

        Args:
            params: the parameter set to test and calculate the cross validated loss for
            model: the model
            x_train: training data
            y_train: training data labels
            scoring_fn: the scoring function to use (can be from 'f1_macro', 'log_loss', 'auprc', or 'brier_score')
            ids: list of ints, the same length as x_train, which holds information on which CV fold each row should be
            assigned to
            nfolds: number of folds to use in cross validation
            verbose: boolean, giving user preference of whether to print information as the trials are being run

        Returns:
            Loss associated with the given parameters, which is to be minimised over time.

        """

        model_copy = model.set_params(**params)

        cv_results = []
        for k in range(nfolds):
            x_train_cv = x_train[ids != k]
            y_train_cv = y_train[ids != k]
            x_test_cv = x_train[ids == k]
            y_test_cv = y_train[ids == k]

            model_copy = model_copy.fit(x_train_cv, y_train_cv)

            y_proba_preds = model_copy.predict_proba(x_test_cv)
            y_proba_preds = np.clip(y_proba_preds, 1e-5, 1 - 1e-5)

            if scoring_fn == 'f1_macro':
                loss = -F1_Macro().calculate(y_test_cv, y_proba_preds)
            elif scoring_fn == 'log_loss':
                loss = LogLoss().calculate(y_test_cv, y_proba_preds)
            elif scoring_fn == 'brier_score':
                loss = BrierScore().calculate(y_test_cv, y_proba_preds)
            elif scoring_fn == 'auprc':
                loss = -AUPRC().calculate(y_test_cv, y_proba_preds)
            elif scoring_fn == 'auroc':
                loss = -AUROC().calculate(y_test_cv, y_proba_preds)
            else:
                raise NotImplementedError(
                    'scoring_fn should be one of ''f1_macro'', ''log_loss'', ''auprc'', ''auroc''or ''brier_score''. ' +
                    '{} given'.format(scoring_fn))

            cv_results.append(loss)

        to_minimise = np.mean(cv_results)
        if verbose:
            print(params)
            print('Loss: {}'.format(to_minimise))

        return to_minimise


class TunerInterface(ComponentInterface):

    registered_flavors = {
        'RandomSearchTuner': RandomSearchTuner,
        'BayesianTuner': BayesianTuner,
    }
