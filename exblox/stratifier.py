from abc import abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Any, Dict, List, Tuple
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface
from .dataset import DataSet


class Stratifier(ConfigurableComponent):
    """
    Stratify a dataset, saving partition indexes.

    `Stratifier` is an abstract base class that must be subclassed, implementing a `partition_data` method.
    """

    def __init__(self, config: Dict, partition_idxs: List[Tuple[List[int], List[int]]] = None):
        super().__init__(config)
        self.validate_config(config)
        self.partition_idxs = partition_idxs

    def stratify(self, data_set: DataSet):
        self.partition_idxs = self.partition_data(data_set)

    @property
    def is_stratified(self):
        return self.partition_idxs is not None

    @property
    def n_partitions(self):
        return len(self.partition_idxs)

    @abstractmethod
    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        """
        Stratify a `DataSet`, yielding a list of partition indexes  in the format (x, y).

        Args:
            data_set: The `DataSet` to stratify.

        Returns: A list of partition indexes in the format (x, y).

        """
        pass

    @classmethod
    @abstractmethod
    def validate_config(cls, config):
        """
        Confirm that the `Stratifier` config contains the correct information

        Args:
            config: A `Stratifer` config.

        Raises: ValueError if the config is not correct.

        """
        pass

    def materialize_partition(self, partition_id: int, data_set: DataSet) -> Tuple[pd.DataFrame, pd.Series,
                                                                                   pd.DataFrame, pd.Series]:
        """
        Create training and testing dataset based on the partition, which indicate the ids for the test set.

        Args:
            partition_id: Index of the partition within self.partition_idxs.

        Returns: X_train, y_train, X_test, y_test
        """
        train_partition_ids, test_partition_ids = self.partition_idxs[partition_id]
        x_train = data_set.x.iloc[train_partition_ids]
        y_train = data_set.y.iloc[train_partition_ids]
        x_test = data_set.x.iloc[test_partition_ids]
        y_test = data_set.y.iloc[test_partition_ids]
        return x_train, y_train, x_test, y_test


class PartitionedLabelStratifier(Stratifier):

    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        """Randomly shuffle and split the doc_list into n_partitions roughly equal lists, stratified by label."""
        label_list = data_set.y
        skf = StratifiedKFold(n_splits=self.config['n_partitions'], random_state=42, shuffle=True)
        x = np.zeros(len(label_list))  # split takes a X argument for backwards compatibility and is not used
        partition_indexes = skf.split(x, label_list)
        partitions = []
        for p_id, p in enumerate(partition_indexes):
            partitions.append(p)
        return partitions

    @classmethod
    def validate_config(cls, config):
        for key in ['n_partitions', ]:
            if key not in config:
                raise ValueError(f"{cls.__name__} config must contain entry '{key}'.")
        return True


class TrainTestStratifier(Stratifier):

    def __init__(self, config: Dict, partition_idxs=None):
        super().__init__(config, partition_idxs)
        self.test_split_size = config['test_split_size']

    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        """Split data once into train and test sets. Percentage of data in test set supplied as argument."""
        df_len = len(data_set.x.index)
        perm = np.random.permutation(df_len)
        train_end = int((1-self.test_split_size) * df_len)
        train_idx = perm[:train_end]
        test_idx = perm[train_end:]
        partitions = [(train_idx, test_idx)]
        return partitions

    @classmethod
    def validate_config(cls, config):
        for key in ['test_split_size', ]:
            if key not in config:
                raise ValueError(f"{cls.__name__} config must contain entry '{key}'.")
        return True


class StratifierInterface(ComponentInterface):

    registered_flavors = {
        'PartitionedLabelStratifier': PartitionedLabelStratifier,
        'TrainTestStratifier': TrainTestStratifier,
    }

    serialization_schema = {
        'partition_idxs': {'required': False, }
    }

    @classmethod
    def additional_info_for_deserialization(cls, d: Dict) -> Dict[str, Any]:
        partition_idxs = d['partition_idxs']
        return {'partition_idxs': partition_idxs}

    @classmethod
    def additional_info_for_serialization(cls, stratifier: Stratifier) -> Dict[str, Any]:
        return {'partition_idxs': stratifier.partition_idxs}
