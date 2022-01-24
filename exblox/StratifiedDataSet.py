import pandas as pd
from .dataset import DataSet, DataSetInterface
from .stratifier import Stratifier
from typing import Dict, Tuple


class StratifiedDataSet(DataSet):
    """Combine a DataSet and a Stratifier in a single iterable object that returns data partitions as defined by the
    stratifier."""

    def __init__(self, data_set: DataSet, stratifier: Stratifier):
        super().__init__(data_set.config, data_set.data)
        self.stratifier = stratifier
        if not stratifier.is_stratified:
            # stratifier may already contain partition indexes if it's being deserialized, in which case we don't want
            #  to overwrite the partitions indexes with new ones
            self.stratifier.stratify(self)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Iterate over the partitions.

        Returns: X_train, y_train, X_test, y_test
        """
        if self.n < self.stratifier.n_partitions:
            return_value = self.stratifier.materialize_partition(self.n, self)
            self.n += 1
            return return_value
        else:
            raise StopIteration


class StratifiedDataSetInterface(DataSetInterface):

    registered_flavors = {}

    @classmethod
    def serialize(cls, component) -> Dict:
        """
        Serialize a StratifiedDataSet into a DataSet. (`StratifiedDataSet`'s `Stratifier` should be serialized
        independently.)

        Args:
            component: A `StratifiedDataSet`.

        Returns: A serialization dictionary for `DataSet`, with keys `flavor` (DataSet), `config`, and `data`.

        """
        d = {
            # Over-ride the flavor to be `DataSet`, not `StratifiedDataSet`, so that it doesn't try to deserialize a
            # `StratifiedDataSet` directly (but rather `DataSet` and `Stratifier` separately).
            'flavor': DataSet.__name__,
            'config': component.config,
        }
        more_info = cls.additional_info_for_serialization(component)
        d = {**d, **more_info}
        cls.validate_serialization_config(d)
        return d

    @classmethod
    def configure(cls, d: Dict, **kwargs) -> None:
        raise NotImplementedError('`configure` is not implemented for `StratifiedDataSetInterface`. Configure a'
                                  '`DataSet` and `Stratifier` separately and assemble them into a `StratifiedDataSet`.')

    @classmethod
    def deserialize(cls, d: Dict) -> None:
        raise NotImplementedError('`configure` is not implemented for `StratifiedDataSetInterface`. Configure a'
                                  '`DataSet` and `Stratifier` separately and assemble them into a `StratifiedDataSet`.')
