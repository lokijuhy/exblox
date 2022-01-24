import unittest
import pandas as pd
import numpy as np
import random

import sklearn.preprocessing
import skorch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from torch import nn
import torch
from skorch import NeuralNet
from exblox.architecture import ArchitectureInterface, PipelineInterface, ColumnTransformerInterface,\
    FunctionTransformerInterface, SkorchNeuralNetInterface
from exblox.architecture import get_instance_import_path, get_func_or_class_import_path
from exblox.architectures.MLP import MLP
from exblox.utilities.utilities import tofloat32


def get_test_data_set():
    column_names = ['A', 'B', 'C', 'D']
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(column_names))), columns=column_names)
    df['label'] = np.where(df[column_names[0]] > 50, 1, 0)
    return df


class TestGetFuncOrClassImportPath(unittest.TestCase):

    def test_get_func_import_path(self):
        result = get_func_or_class_import_path(tofloat32)
        expected_result = 'exblox.utilities.utilities.tofloat32'
        self.assertEqual(result, expected_result)

    def test_get_class_import_path(self):
        result = get_func_or_class_import_path(torch.optim.SGD)
        expected_result = 'torch.optim.sgd.SGD'
        self.assertEqual(result, expected_result)


class TestGetInstanceImportPath(unittest.TestCase):

    def test_get_instance_import_path(self):
        mlp_instance = MLP(10)
        result = get_instance_import_path(mlp_instance)
        expected_result = 'exblox.architectures.MLP.MLP'
        self.assertEqual(result, expected_result)


class TestArchitectureInterface(unittest.TestCase):

    def test_pipeline_configure(self):
        config = {
            'flavor': 'sklearn.pipeline.Pipeline',
            'config': {
                'steps': [
                    {
                        'flavor': 'sklearn.preprocessing.StandardScaler',
                        'config': {
                            'with_mean': True,
                        }
                    },
                    {
                        'flavor': 'sklearn.svm.SVC',
                        'config': {

                        }
                    },
                ],
            },
        }
        pipe = ArchitectureInterface.configure(config)

    def test_nested_pipeline_configure(self):
        one_pipe_config = {
            'flavor': 'sklearn.compose.ColumnTransformer',
            'name': 'cyc',
            'config': {
                'steps': [
                    {
                        'flavor': 'sklearn.preprocessing.StandardScaler',
                        'name': 'scaler',
                        'args': {'columns': ['A', 'B']},
                        'config': {'with_mean': True, },
                    },
                    {
                        'flavor': 'sklearn.svm.SVC',
                        'name': 'svc',
                        'args': {'columns': ['C', 'D']},
                        'config': {},
                    },
                ],
            },
        }
        nested_config = {
            'flavor': 'sklearn.pipeline.Pipeline',
            'config': {
                'steps':
                    [
                        one_pipe_config,
                    ],
            },
        }
        pipe = ArchitectureInterface.configure(nested_config)

    def test_pipeline_serialize(self):
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

        expected_serialization = {
            'flavor': 'sklearn.pipeline.Pipeline',
            'config': {
                'steps': [
                    {
                        'flavor': 'sklearn.preprocessing._data.StandardScaler',
                        'name': 'scaler',
                        'config': {
                            'copy': True,
                            'with_mean': True,
                            'with_std': True
                        }
                    },
                    {
                        'flavor': 'sklearn.svm._classes.SVC',
                        'name': 'svc',
                        'config': {
                            'C': 1.0,
                            'break_ties': False,
                            'cache_size': 200,
                            'class_weight': None,
                            'coef0': 0.0,
                            'decision_function_shape': 'ovr',
                            'degree': 3,
                            'gamma': 'scale',
                            'kernel': 'rbf',
                            'max_iter': -1,
                            'probability': False,
                            'random_state': None,
                            'shrinking': True,
                            'tol': 0.001,
                            'verbose': False
                        }
                    },
                ],
            },
        }
        serialization = ArchitectureInterface.serialize(pipe)
        self.assertEqual(serialization, expected_serialization)

    def test_nested_pipeline_serialize(self):
        column_names = ['A', 'B', 'C', 'D']
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(column_names))), columns=column_names)
        df['E'] = random.choices(['one', 'two', 'three'], k=100)
        df['label'] = np.where(df[column_names[0]] > 50, 1, 0)

        categorical_columns = ['E']
        numerical_columns = ['A', 'B', 'C', 'D']

        categorical_encoder = OneHotEncoder(handle_unknown='ignore')

        numerical_pipe = Pipeline([
            ('scaler', StandardScaler())
        ])

        preprocessing = ColumnTransformer(
            [('cat', categorical_encoder, categorical_columns),
             ('num', numerical_pipe, numerical_columns)
             ])

        lr_pipe = Pipeline([
            ('preprocess', preprocessing),
            ('classifier',
             LogisticRegression(random_state=94, penalty='l1', solver='liblinear', class_weight='balanced'))
        ])

        # x = df.copy().drop('label', axis=1)
        # y = df['label']
        # lr_pipe.fit(x, y)

        expected_serialization = {
            'flavor': 'sklearn.pipeline.Pipeline',
            'config': {
                'steps': [
                    {
                        'flavor': 'sklearn.compose.ColumnTransformer',
                        'name': 'preprocess',
                        'config': {
                            'steps': [
                                {
                                    'flavor': 'sklearn.preprocessing._encoders.OneHotEncoder',
                                    'name': 'cat',
                                    'args': {'columns': ['E']},
                                    'config': {
                                        'categories': 'auto',
                                        'drop': None,
                                        # 'dtype': 'numpy.float64',  # TODO: this is a problem
                                        'handle_unknown': 'ignore',
                                        'sparse': True
                                    },
                                },
                                {
                                    'flavor': 'sklearn.pipeline.Pipeline',
                                    'name': 'num',
                                    'args': {'columns': ['A', 'B', 'C', 'D']},
                                    'config': {
                                        'steps': [
                                            {
                                                'flavor': 'sklearn.preprocessing._data.StandardScaler',
                                                'name': 'scaler',
                                                'config': {
                                                    'copy': True,
                                                    'with_mean': True,
                                                    'with_std': True
                                                },
                                            }
                                        ]
                                    },
                                }
                            ]
                        },
                    },
                    {
                        'flavor': 'sklearn.linear_model._logistic.LogisticRegression',
                        'name': 'classifier',
                        'config': {
                            'C': 1.0,
                            'class_weight': 'balanced',
                            'dual': False,
                            'fit_intercept': True,
                            'intercept_scaling': 1,
                            'l1_ratio': None,
                            'max_iter': 100,
                            'multi_class': 'auto',
                            'n_jobs': None,
                            'penalty': 'l1',
                            'random_state': 94,
                            'solver': 'liblinear',
                            'tol': 0.0001,
                            'verbose': 0,
                            'warm_start': False
                        },
                    }
                ]
            },
        }
        serialization = ArchitectureInterface.serialize(lr_pipe)
        self.assertEqual(serialization, expected_serialization)

    def test_nested_pipeline_serialize_deserialize(self):
        column_names = ['A', 'B', 'C', 'D']
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(column_names))), columns=column_names)
        df['E'] = random.choices(['one', 'two', 'three'], k=100)
        df['label'] = np.where(df[column_names[0]] > 50, 1, 0)
        x = df.copy().drop('label', axis=1)
        y = df['label']

        categorical_columns = ['E']
        numerical_columns = ['A', 'B', 'C', 'D']

        categorical_encoder = OneHotEncoder(handle_unknown='ignore')  # 0a

        numerical_pipe = Pipeline([  # 0b
            ('scaler', StandardScaler())  # 0ba
        ])

        preprocessing = ColumnTransformer(
            [('cat', categorical_encoder, categorical_columns),  # 0a
             ('num', numerical_pipe, numerical_columns)  # 0b
             ])

        log_reg = LogisticRegression(random_state=94, penalty='l1', solver='liblinear', class_weight='balanced')
        pipe = Pipeline([
            ('preprocess', preprocessing),  # 0
            ('classifier', log_reg),  # 1

        ])

        serialization = ArchitectureInterface.serialize(pipe)
        recreated_pipe = ArchitectureInterface.deserialize(serialization)

        self.assertEqual(type(recreated_pipe), Pipeline)
        self.assertEqual(len(recreated_pipe.steps), 2)

        step_0_name, step_0_obj = recreated_pipe.steps[0]
        self.assertEqual(step_0_name, 'preprocess')
        self.assertEqual(type(step_0_obj), ColumnTransformer)

        step_0a_name, step_0a_obj, step_0a_cols = step_0_obj.transformers[0]
        self.assertEqual(step_0a_name, 'cat')
        self.assertEqual(type(step_0a_obj), OneHotEncoder)
        self.assertEqual(step_0a_cols, categorical_columns)
        self.assertEqual(step_0a_obj.get_params(), categorical_encoder.get_params())

        step_0b_name, step_0b_obj, step_0b_cols = step_0_obj.transformers[1]
        self.assertEqual(step_0b_name, 'num')
        self.assertEqual(type(step_0b_obj), Pipeline)
        self.assertEqual(step_0b_cols, numerical_columns)

        step_0ba_name, step_0ba_obj = step_0b_obj.steps[0]
        self.assertEqual(step_0ba_name, 'scaler')
        self.assertEqual(type(step_0ba_obj), StandardScaler)

        step_1_name, step_1_obj = recreated_pipe.steps[1]
        self.assertEqual(step_1_name, 'classifier')
        self.assertEqual(type(step_1_obj), LogisticRegression)
        self.assertEqual(step_1_obj.get_params(), log_reg.get_params())

        pipe.fit(x, y)
        original_pipe_result = pipe.predict(x)

        recreated_pipe.fit(x, y)
        recreated_pipe_result = recreated_pipe.predict(x)

        np.testing.assert_array_almost_equal(recreated_pipe_result, original_pipe_result)

    def test_skorch_within_pipeline(self):
        column_names = ['A', 'B', 'C', 'D']
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(column_names))), columns=column_names)
        df['E'] = random.choices(['one', 'two', 'three'], k=100)
        df['label'] = np.where(df[column_names[0]] > 50, 1, 0)

        categorical_columns = ['E']
        numerical_columns = ['A', 'B', 'C', 'D']

        categorical_encoder = OneHotEncoder(handle_unknown='ignore')

        numerical_pipe = Pipeline([  # 0b
            ('scaler', StandardScaler())  # 0ba
        ])

        preprocessing = ColumnTransformer(  # 0
            [('cat', categorical_encoder, categorical_columns),  # 0a
             ('num', numerical_pipe, numerical_columns)  # 0b
             ]
        )

        net = NeuralNet(  # 2
            MLP(input_layer_size=4, hidden_layer_size=20, dropout_p=0),
            criterion=nn.BCELoss,
            # criterion__weight=torch.tensor(0.17),
            lr=0.01,
            optimizer=torch.optim.SGD,
            batch_size=32,
            max_epochs=100,
            verbose=0,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
        )

        func_transformer = FunctionTransformer(tofloat32, accept_sparse=True)  # 1

        nn_pipe = Pipeline([
            ('preprocess', preprocessing),    # 0
            ('tofloat32', func_transformer),  # 1
            ('classifier', net)               # 2
        ])

        serialization = ArchitectureInterface.serialize(nn_pipe)
        recreated_pipe = ArchitectureInterface.deserialize(serialization)

        self.assertEqual(type(recreated_pipe), Pipeline)
        self.assertEqual(len(recreated_pipe.steps), 3)

        step_0_name, step_0_obj = recreated_pipe.steps[0]
        self.assertEqual(step_0_name, 'preprocess')
        self.assertEqual(type(step_0_obj), ColumnTransformer)

        step_0a_name, step_0a_obj, step_0a_cols = step_0_obj.transformers[0]
        self.assertEqual(step_0a_name, 'cat')
        self.assertEqual(type(step_0a_obj), OneHotEncoder)
        self.assertEqual(step_0a_cols, categorical_columns)
        self.assertEqual(step_0a_obj.get_params(), categorical_encoder.get_params())

        step_0b_name, step_0b_obj, step_0b_cols = step_0_obj.transformers[1]
        self.assertEqual(step_0b_name, 'num')
        self.assertEqual(type(step_0b_obj), Pipeline)
        self.assertEqual(step_0b_cols, numerical_columns)

        step_0ba_name, step_0ba_obj = step_0b_obj.steps[0]
        self.assertEqual(step_0ba_name, 'scaler')
        self.assertEqual(type(step_0ba_obj), StandardScaler)

        step_1_name, step_1_obj = recreated_pipe.steps[1]
        self.assertEqual(step_1_name, 'tofloat32')
        self.assertEqual(type(step_1_obj), FunctionTransformer)
        self.assertEqual(step_1_obj.get_params(), func_transformer.get_params())

        step_2_name, step_2_obj = recreated_pipe.steps[2]
        self.assertEqual(step_2_name, 'classifier')
        self.assertEqual(type(step_2_obj), NeuralNet)

        step_2_params = step_2_obj.get_params()
        net_params = net.get_params()

        # only check parameters with core data types- ignore NeuralNet parameters of type `object` and `class`
        core_types = (str, int, float, bool, dict)
        step_2_filtered_params = {k: v for k, v in step_2_params.items() if isinstance(v, core_types)}
        net_filtered_params = {k: v for k, v in net_params.items() if isinstance(v, core_types)}
        self.assertEqual(step_2_filtered_params, net_filtered_params)


class TestSkorchNeuralNetInterface(unittest.TestCase):

    def test_configure(self):
        skorch_config = {
            'flavor': 'skorch.NeuralNet',
            'config': {
                'lr': 0.01,
                'batch_size': 32,
                'max_epochs': 100,
                'verbose': 0,
                'iterator_train__shuffle': True,
            },
            'args': {
                'module': {
                    'flavor': 'exblox.architectures.MLP.MLP',
                    'config': {
                        'input_layer_size': 10,
                        'hidden_layer_size': 20,
                        'dropout_p': 0},
                },
                'criterion': {
                    'flavor': 'torch.nn.BCELoss',
                    'instantiate': False,
                },
                'optimizer': {
                    'flavor': 'torch.optim.SGD',
                    'instantiate': False,
                }
            },
        }
        configured_net = SkorchNeuralNetInterface.configure(skorch_config)
        self.assertTrue(isinstance(configured_net, skorch.NeuralNet))
        self.assertTrue(isinstance(configured_net.module, MLP))
        self.assertTrue(configured_net.criterion, torch.nn.BCELoss)
        self.assertTrue(configured_net.optimizer, torch.optim.SGD)
        self.assertEqual(configured_net.batch_size, skorch_config['config']['batch_size'])

    def test_serialize(self):
        net = NeuralNet(
            MLP(input_layer_size=10, hidden_layer_size=20, dropout_p=0),
            criterion=nn.BCELoss,
            lr=0.01,
            optimizer=torch.optim.SGD,
            batch_size=32,
            max_epochs=100,
            verbose=0,
            iterator_train__shuffle=True,
        )
        expected_config = {
            'flavor': 'skorch.NeuralNet',
            'config': {
                'lr': 0.01,
                'batch_size': 32,
                'max_epochs': 100,
                'verbose': 0,
                'iterator_train__shuffle': True,
            },
            'args': {
                'module': {
                    'flavor': 'exblox.architectures.MLP.MLP',
                    'config': {
                        'input_layer_size': 10,
                        'hidden_layer_size': 20,
                        'dropout_p': 0},
                },
                'criterion': {
                    'flavor': 'torch.nn.modules.loss.BCELoss',
                    'instantiate': False,
                },
                'optimizer': {
                    'flavor': 'torch.optim.sgd.SGD',
                    'instantiate': False,
                }
            },
        }
        serialized_net = SkorchNeuralNetInterface.serialize(net)
        self.assertEqual(serialized_net['flavor'], expected_config['flavor'])
        self.assertEqual(serialized_net['args'], expected_config['args'])
        # check that all specific config params are there, but ignore extra params
        for key in expected_config['config']:
            self.assertEqual(serialized_net['config'][key], expected_config['config'][key])

    def test_serialize_deserialize(self):
        net = NeuralNet(
            MLP(input_layer_size=10, hidden_layer_size=20, dropout_p=0),
            criterion=nn.BCELoss,
            lr=0.01,
            optimizer=torch.optim.SGD,
            batch_size=32,
            max_epochs=100,
            verbose=0,
            iterator_train__shuffle=True,
        )
        serialized_net = SkorchNeuralNetInterface.serialize(net)
        configured_net = SkorchNeuralNetInterface.deserialize(serialized_net)
        self.assertTrue(isinstance(configured_net, skorch.NeuralNet))
        self.assertTrue(isinstance(configured_net.module, MLP))
        self.assertTrue(configured_net.criterion, torch.nn.BCELoss)
        self.assertTrue(configured_net.optimizer, torch.optim.SGD)
        self.assertEqual(configured_net.batch_size, 32)


class TestFunctionTransformerInterface(unittest.TestCase):

    def test_configure(self):
        config = {
            'flavor': 'sklearn.preprocessing.FunctionTransformer',
            'args': {
                'function':
                    {
                        'flavor': 'exblox.utilities.utilities.tofloat32',
                        'instantiate': False,
                    },
            },
            'config': {
                'accept_sparse': True
            }
        }
        configured_func_transformer = FunctionTransformerInterface.configure(config)
        self.assertEqual(configured_func_transformer.func, tofloat32)
        params = configured_func_transformer.get_params()
        self.assertEqual(params['accept_sparse'], config['config']['accept_sparse'])

    def test_serialize(self):
        transformer = FunctionTransformer(tofloat32, accept_sparse=True)
        expected_config = {
            'flavor': 'sklearn.preprocessing.FunctionTransformer',
            'args': {
                'function':
                    {
                        'flavor': 'exblox.utilities.utilities.tofloat32',
                        'instantiate': False,
                    },
            },
            'config': {
                'accept_sparse': True
            }
        }
        serialized_func_transformer = FunctionTransformerInterface.serialize(transformer)
        self.assertEqual(serialized_func_transformer['flavor'], expected_config['flavor'])
        self.assertEqual(serialized_func_transformer['args'], expected_config['args'])
        # check that all specific config params are there, but ignore extra params
        for key in expected_config['config']:
            self.assertEqual(serialized_func_transformer['config'][key], expected_config['config'][key])

    def test_serialize_deserialize(self):
        transformer = FunctionTransformer(tofloat32, accept_sparse=True)
        serialized_func_transformer = FunctionTransformerInterface.serialize(transformer)
        configured_func_transformer = FunctionTransformerInterface.deserialize(serialized_func_transformer)
        self.assertEqual(configured_func_transformer.func, tofloat32)
        params = configured_func_transformer.get_params()
        self.assertEqual(params['accept_sparse'], True)
