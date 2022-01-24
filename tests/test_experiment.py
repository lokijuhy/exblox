import unittest
from datetime import datetime
from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from exblox.experiment import Experiment
from exblox.stratifier import TrainTestStratifier
from exblox.StratifiedDataSet import StratifiedDataSet
from exblox.trainer import Trainer
from exblox.tuner import RandomSearchTuner
from exblox.metric import Metric
from exblox.predictor import Predictor


def get_test_data_set():
    column_names = ['A', 'B', 'C', 'D']
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(column_names))), columns=column_names)
    df['label'] = np.where(df[column_names[0]] > 50, 1, 0)
    return df


class TestExperiment(unittest.TestCase):

    def setUp(self):
        self.df = get_test_data_set()
        self.configuration = {
            'DataSet': {
                'flavor': 'DataSet',
                'config': {
                    'features': ['A', 'B', 'C', 'D'],
                    'target': 'label',
                },
            },
            'Stratifier': {
                'flavor': 'TrainTestStratifier',
                'config': {
                    'test_split_size': 0.3,
                },
            },
            'Architecture': {
                'flavor': 'RandomForestClassifier',
                'config': {}
            },
            'Trainer': {
                'flavor': 'Trainer',
                'config': {}
            },
            'Tuner': {
                'flavor': 'RandomSearchTuner',
                'config': {
                    'hyperparameters': {
                        'n_estimators': range(200, 2000, 10),
                        'max_features': ['auto', 'sqrt'],
                        'max_depth': range(10, 110, 11),
                        'min_samples_split': [2, 4, 6, 8, 10],
                        'min_samples_leaf': [1, 2, 5, 10],
                        'bootstrap': [True, False],
                    },
                    'num_cv_folds': 3,
                    'num_iters': 5,
                    'scoring_function': 'f1_macro',
                    'verbose': 1,
                },
            },
            'Metrics': [
                {'flavor': 'F1_Macro', 'config': {'classification_cutoff': 0.5}},
                {'flavor': 'AUPRC', 'config': {}},
                {'flavor': 'AUROC', 'config': {}},
                {'flavor': 'LogLoss', 'config': {}},
            ]
        }
        self.configuration_without_tuner = {
            'DataSet': {
                'flavor': 'DataSet',
                'config': {
                    'features': ['A', 'B', 'C', 'D'],
                    'target': 'label',
                },
            },
            'Stratifier': {
                'flavor': 'TrainTestStratifier',
                'config': {
                    'test_split_size': 0.3,
                },
            },
            'Architecture': {
                'flavor': 'RandomForestClassifier',
                'config': {}
            },
            'Trainer': {
                'flavor': 'Trainer',
                'config': {}
            },
            'Metrics': [
                {'flavor': 'F1_Macro', 'config': {'classification_cutoff': 0.5}},
                {'flavor': 'AUPRC', 'config': {}},
                {'flavor': 'AUROC', 'config': {}},
                {'flavor': 'LogLoss', 'config': {}},
            ]
        }

    def test_configure(self):
        exp = Experiment.configure(config=self.configuration, data=self.df)

        self.assertTrue(isinstance(exp.stratified_dataset, StratifiedDataSet))
        self.assertEqual(exp.stratified_dataset.config, self.configuration['DataSet']['config'])
        pd.testing.assert_frame_equal(exp.stratified_dataset.data, self.df)

        self.assertTrue(isinstance(exp.stratified_dataset.stratifier, TrainTestStratifier))
        self.assertEqual(exp.stratified_dataset.stratifier.config, self.configuration['Stratifier']['config'])

        self.assertTrue(isinstance(exp.trainer.architecture, RandomForestClassifier))
        # self.assertEqual(exp.dataset.config, self.configuration['DataSet']['config'])  # TODO How to test this?

        self.assertTrue(isinstance(exp.trainer, Trainer))
        self.assertEqual(exp.trainer.config, self.configuration['Trainer']['config'])

        self.assertTrue(isinstance(exp.trainer.tuner, RandomSearchTuner))
        self.assertEqual(exp.trainer.tuner.config, self.configuration['Tuner']['config'])

        self.assertTrue(isinstance(exp.metrics, list))
        for i, metric in enumerate(exp.metrics):
            self.assertTrue(isinstance(metric, Metric))
            self.assertEqual(metric.config, self.configuration['Metrics'][i]['config'])

    def test_serialize_before_go(self):
        exp = Experiment.configure(config=self.configuration, data=self.df)
        exp_dict = exp.serialize()
        for expected_key in ['metadata', 'components', 'results']:
            self.assertTrue(expected_key in exp_dict)

        for component in ['DataSet', 'Stratifier', 'Architecture', 'Trainer', 'Tuner', 'Metrics']:
            component_dict = exp_dict['components'][component]
            orig_component_dict = self.configuration[component]

            if component == 'Architecture':
                continue  # see test_architecture.py

            if component == 'Metrics':
                for i, m in enumerate(component_dict):
                    for key in ['flavor', 'config']:
                        self.assertEqual(component_dict[i][key], orig_component_dict[i][key])
            else:
                for key in ['flavor', 'config']:
                    self.assertEqual(component_dict[key], orig_component_dict[key])

        # results
        self.assertTrue('partition_predictors' in exp_dict['results'].keys())
        self.assertTrue(isinstance(exp_dict['results']['partition_predictors'], list))
        self.assertTrue(all([isinstance(p, Predictor) for p in exp_dict['results']['partition_predictors']]))

        self.assertTrue('evaluation' in exp_dict['results'].keys())
        self.assertTrue(isinstance(exp_dict['results']['evaluation'], dict))

        self.assertTrue('final_predictor' in exp_dict['results'].keys())
        self.assertIsNone(exp_dict['results']['final_predictor'])

    def test_deserialize(self):
        exp = Experiment.configure(config=self.configuration, data=self.df)
        exp_dict = exp.serialize()
        exp_deserialized = Experiment.deserialize(exp_dict)

        # DataSet
        self.assertEqual(type(exp_deserialized.stratified_dataset), type(exp.stratified_dataset))
        pd.testing.assert_frame_equal(exp_deserialized.stratified_dataset.data, exp.stratified_dataset.data)

        # Stratifier
        self.assertEqual(type(exp_deserialized.stratified_dataset.stratifier), type(exp.stratified_dataset.stratifier))
        self.assertEqual(exp_deserialized.stratified_dataset.stratifier.config, exp.stratified_dataset.stratifier.config)
        for p, partition in enumerate(exp.stratified_dataset.stratifier.partition_idxs):
            for d, indx in enumerate(partition):
                np.testing.assert_almost_equal(exp_deserialized.stratified_dataset.stratifier.partition_idxs[p][d],
                                               exp.stratified_dataset.stratifier.partition_idxs[p][d])
                # for i, v in enumerate(indx):
                #     self.assertEqual(exp_deserialized.stratifier.partition_idxs[p][d][i],
                #                      exp.stratifier.partition_idxs[p][d][i])

        # Trainer
        self.assertEqual(type(exp_deserialized.trainer), type(exp.trainer))
        self.assertEqual(exp_deserialized.trainer.config, exp.trainer.config)

        # Architecture
        # see test_architecture.py

        # Tuner
        self.assertEqual(type(exp_deserialized.trainer.tuner), type(exp.trainer.tuner))
        self.assertEqual(exp_deserialized.trainer.tuner.config, exp.trainer.tuner.config)

        # Metrics
        for i, metric in enumerate(exp.metrics):
            self.assertEqual(type(exp_deserialized.metrics[i]), type(exp.metrics[i]))
            self.assertEqual(exp_deserialized.metrics[i].config, exp.metrics[i].config)

        # Metadata
        self.assertTrue(isinstance(exp_deserialized.metadata, dict))
        self.assertEqual(len(exp_deserialized.metadata), 0)  # dictionary is empty if not `go` not called

    def test_serialize_deserialize_without_Tuner(self):
        exp = Experiment.configure(config=self.configuration_without_tuner, data=self.df)
        exp_dict = exp.serialize()

        self.assertTrue('Tuner' not in exp_dict['components'])
        exp_deserialized = Experiment.deserialize(exp_dict)

        # Tuner
        self.assertIsNone(exp_deserialized.trainer.tuner)

    def test_deserialize_with_metadata(self):
        metadata = {'name': 'test'}
        config = deepcopy(self.configuration)
        config['metadata'] = metadata

        exp = Experiment.configure(config=config, data=self.df)
        exp_dict = exp.serialize()
        exp_deserialized = Experiment.deserialize(exp_dict)

        self.assertTrue(isinstance(exp_deserialized.metadata, dict))
        for key, value in metadata.items():
            self.assertEqual(exp_deserialized.metadata[key], value)

    def test_deserialize_after_go(self):
        exp = Experiment.configure(config=self.configuration, data=self.df)
        exp.go()
        exp_dict = exp.serialize()
        exp_deserialized = Experiment.deserialize(exp_dict)

        # DataSet
        self.assertEqual(type(exp_deserialized.stratified_dataset), type(exp.stratified_dataset))
        pd.testing.assert_frame_equal(exp_deserialized.stratified_dataset.data, exp.stratified_dataset.data)

        # Stratifier
        self.assertEqual(type(exp_deserialized.stratified_dataset.stratifier), type(exp.stratified_dataset.stratifier))
        self.assertEqual(exp_deserialized.stratified_dataset.stratifier.config, exp.stratified_dataset.stratifier.config)
        for p, partition in enumerate(exp.stratified_dataset.stratifier.partition_idxs):
            for d, indx in enumerate(partition):
                np.testing.assert_almost_equal(exp_deserialized.stratified_dataset.stratifier.partition_idxs[p][d],
                                               exp.stratified_dataset.stratifier.partition_idxs[p][d])
                # for i, v in enumerate(indx):
                #     self.assertEqual(exp_deserialized.stratifier.partition_idxs[p][d][i],
                #                      exp.stratifier.partition_idxs[p][d][i])

        # Trainer
        self.assertEqual(type(exp_deserialized.trainer), type(exp.trainer))
        self.assertEqual(exp_deserialized.trainer.config, exp.trainer.config)

        # Architecture
        # see test_architecture.py

        # Tuner
        self.assertEqual(type(exp_deserialized.trainer.tuner), type(exp.trainer.tuner))
        self.assertEqual(exp_deserialized.trainer.tuner.config, exp.trainer.tuner.config)

        # Metrics
        for i, metric in enumerate(exp.metrics):
            self.assertEqual(type(exp_deserialized.metrics[i]), type(exp.metrics[i]))
            self.assertEqual(exp_deserialized.metrics[i].config, exp.metrics[i].config)

        # Metadata
        self.assertEqual(exp_deserialized.metadata, exp.metadata)
        self.assertTrue(isinstance(exp_deserialized.metadata['run_date'], datetime))

        # Evaluation
        pd.testing.assert_frame_equal(exp_deserialized.evaluation, exp.evaluation)

        # Partition Predictors
        # assert that for every partition, the predictors predict the same values
        for partition_idx in range(exp.stratified_dataset.stratifier.n_partitions):
            x_train, y_train, x_test, y_test = exp.stratified_dataset.stratifier.materialize_partition(partition_idx, exp.stratified_dataset)
            x_train_des, y_train_des, x_test_des, y_test_des = exp_deserialized.stratified_dataset.stratifier.materialize_partition(partition_idx, exp_deserialized.stratified_dataset)
            # before checking the predictors, make sure the X they're predicting on is the same.
            pd.testing.assert_frame_equal(x_test_des, x_test)
            y_pred = exp.partition_predictors[partition_idx].predict(x_test)
            y_pred_des = exp_deserialized.partition_predictors[partition_idx].predict(x_test_des)
            np.testing.assert_almost_equal(y_pred_des, y_pred)

        # Final Predictor
        self.assertTrue(isinstance(exp_deserialized.final_predictor, Predictor))
        # assert same predictions on sample partition
        x_train, y_train, x_test, y_test = exp.stratified_dataset.stratifier.materialize_partition(0, exp.stratified_dataset)
        x_train_des, y_train_des, x_test_des, y_test_des = exp_deserialized.stratified_dataset.stratifier.materialize_partition(0, exp_deserialized.stratified_dataset)
        # before checking the predictors, make sure the X they're predicting on is the same.
        pd.testing.assert_frame_equal(x_test_des, x_test)
        y_pred = exp.final_predictor.predict(x_test)
        y_pred_des = exp_deserialized.final_predictor.predict(x_test_des)
        np.testing.assert_almost_equal(y_pred_des, y_pred)

        # Partition Training Metadata
        self.assertTrue(isinstance(exp_deserialized.partition_training_metadata, list))
        # TrainTestStratifier has 1 partition
        self.assertEqual(len(exp_deserialized.partition_training_metadata), 1)
        for part_train_metadata in exp_deserialized.partition_training_metadata:
            self.assertTrue(isinstance(part_train_metadata, dict))

        # Final Training Metadata
        self.assertTrue(isinstance(exp_deserialized.final_training_metadata, dict))
