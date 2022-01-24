# ExBlox

ExBlox provides a plug-and-play ML experimentation framework, so you can rapidly iterate and test different models without writing and re-writing the same boilerplate ML code.

# Approach

An ExBlox machine learning `Experiment` is built from 6 components:
1. A `DataSet`, which not only holds the data and keeps track of the feature and target columns.
2. A `Stratifier`, telling the experiment how to split the data into train/test partitions.
3. A model `Architecture` to train.
4. A `Trainer` that controls how the model is trained, for example sklearn or PyTorch. 
5. A `Tuner`, which governs hyperparameter search.
6. A list of `Metrics` that the model is evaluated on.

Each of these _components_ has a few `flavors` you can pick from, for example, you might choose a `TrainTestStratifier` or a `PartitionedLabelStratifier`.
Each component also needs a `config`. For example, for a `TrainTestStratifier`, you'll need to tell it what `test_split_size` to use, or for the `PartitionedLabelStratifier`, the number of partitions.
Once you've selected each component's flavor and configured it, just mix them together in an `Experiment`, and call `go()`!

![ExBlox Class Diagram](ExBloxClassDiagram.svg)

ExBlox provides the following component flavors out of the box (and it's easy to write and re-use your own!):
* **DataSet**: just the base `DataSet`
* **Stratifier**: You can pick from a `TrainTestStratifier` or `PartitionedLabelStratifier` that generates multiple partitions.
* **Architecture**: Any sklearn model works as an `Architecture`, as well as sklearn `Pipelines`. If you prefer PyTorch, create a skorch model.
* **Trainer**: The base `Trainer` works with all sklearn objects. For a skorch Architecture, use `SkorchTrainer`.
* **Tuner**: You can pick from `RandomSearchTuner` or `BayesianTuner`. You can also not include a `Tuner` at all- it's optional.
* **Metric**: `F1_Macro`, `AUPRC`, `AUROC`, `LogLoss`, and `BrierScore` are ready to go.

You can also pass `Experiment` a `metadata` dictionary to store whatever information you want to track.


# Example usage

## Define an `Experiment`
There are 2 ways to define an experiment: via object creation or via a config. 
If you are working in a notebook, you may want to use the object-creation method, and plug-and-play with different experiment components.

Once you have decided on a model definition you are happy with, you might find it helpful to write it as a configuration for readability and re-use.
This approach has the advantage of providing a single point of truth for everything that contributes to the experiment as an easy-to-read configuration. 

### Create an `Experiment` from objects

```python
from dataset import DataSet
from stratifier import TrainTestStratifier
from sklearn.ensemble import RandomForestClassifier
from trainer import Trainer
from tuner import RandomSearchTuner
from metric import F1_Macro, AUPRC
from experiment import Experiment

# =====  1. DATASET =====
df = pd.read_csv(data_path)
dataset_config = {
    'features': ['A', 'B', 'C', 'D'],
    'target': 'E',
}
dataset = DataSet(dataset_config, df)

# =====  2. STRATIFIER =====
stratifier = TrainTestStratifier({'test_split_size': 0.3})

# =====  3. ARCHITECTURE =====
architecture = RandomForestClassifier()

# =====  5. TUNER (optional) =====
tuner_config = {
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
}
tuner = RandomSearchTuner(tuner_config)

# =====  4. TRAINER =====
trainer = Trainer(architecture=architecture, tuner=tuner)

# =====  6. METRICS =====
metrics = [F1_Macro, AUPRC]
metadata = {'name': 'random forest with hyperparam search'}


exp = Experiment(dataset, stratifier, trainer, metrics, metadata)
exp.go()
```

### Create an `Experiment` from configuration
When defining an Experiment via a config, all components and their configurations are written out in the structure below.
The only object that is not configured here is the dataframe- that is passed to Experiment separately.

```python
from experiment import Experiment

config = {
        'DataSet': {
            'flavor': 'DataSet',
            'config': {
                'features': ['A', 'B', 'C', 'D'],
                'target': 'E',
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

exp = Experiment.configure(config, df)
exp.go()
```

## Running an `Experiment`
To run an experiment, just call `experiment.go()`.

After a completed `go()` is complete, `Experiment will contain the following data:
* `Experiment.evaluation`: a dataframe with the evaluations of each partition's model on the specified `metrics`.
* `Experiment.partition_predictors`: The predictors that were trained during each experiment.
* `Experiment.final_predictor`: The predictor that was trained on the full dataset.
* `Experiment.partition_training_metadata`: Metadata collected during the training processes of each partition.
* `Experiment.final_training_metadata`: Metadata collected during the training processes of the `final_predictor`.

## Saving and Loading Experiments
### Save
Before saving the Experiment to a file, it should be serialized. 
This exports the experiment definition to (mostly) a configuration dictionary. 
Serialization enables saved `Experiment`s to be saved and re-opened with different versions of the `Experiment` library,
which means it reduces the risk of your saved Experiment being un-openable in the future.

```python
import pickle

serialized_exp = exp.serialize()
with open(experiment_file_path, "wb+") as f:  # TODO: check syntax
    pickle.dump(serialized_exp, f)

```

### Load
After loading a serialized experiment from a pickle, deserialize it:
```python
with open(experiment_file_path, "rb+") as f:
    serialized_exp = pickle.load(f)
exp = Experiment.deserialize(serialized_exp)
```

## Using an `Experiment` to make novel predictions

```python
# create a new `DataSet` object from your new features_df using the Experiment's dataset configuration.
data_set = DataSet(exp.dataset.config, features_df)
preds_proba = exp.final_predictor.predict_proba(data_set.x)
```

## Appendix

### Configuring an sklearn Pipeline

```yaml
Architecture:
    flavor: Pipeline
    config:
        steps:
          - flavor: sklearn.compose.ColumnTransformer
            name: 'preprocessing'
            config:
                steps:
                    - name: 'scaler'
                      flavor: sklearn.preprocessing.StandardScaler
                      config:
                        with_mean: True
                      args:
                        columns:
                            - 'no_show_before'
                            - 'sched_days_advanced'
                            - 'age'
                            - 'hour_sched'
                            - 'distance_to_usz'
                            - 'month'
                    - name: 'onehot'
                      flavor: sklearn.preprocessing.OneHotEncoder
                      config:
                          handle_unknown: 'ignore'
                      args:
                          columns:
                              - 'marital'
                              - 'modality'
                              - 'day_of_week_str'
          - flavor: LogisticRegression
            name: 'classifier'
            config:
                penalty: 'l1'
                solver: 'liblinear'
                class_weight: 'balanced'
```
