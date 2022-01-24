import pandas as pd
from typing import Any, Dict
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface


class DataSet(ConfigurableComponent):
    """
    A dataset class config dictionary contains the following configurable elements:
    - features: name of the features to be used in building the features tensor
    - target: name to be used in building the target tensor
    """

    def __init__(self, config: Dict, data: pd.DataFrame):
        super().__init__(config)
        self.validate_config(config, data)
        self.data = data
        self.features_list = config['features']
        self.target = config['target']

    @property
    def x(self) -> pd.DataFrame:
        return self.data[self.features_list]

    @property
    def y(self) -> pd.Series:
        return self.data[self.target]

    @staticmethod
    def validate_config(config, data):
        """Make sure the config aligns with the data (referenced columns exist)."""
        for key in ['features', 'target']:
            if key not in config:
                raise ValueError(f"DataSet config must contain entry '{key}'.")

        for col in config['features']:
            if col not in data.columns:
                raise ValueError(f'Feature column {col} not found in dataset.')

        if config['target'] not in data.columns:
            raise ValueError(f"Target column {config['target']} not found in dataset.")

        return True


class DataSetInterface(ComponentInterface):

    registered_flavors = {
        'DataSet': DataSet,
    }

    serialization_schema = {
        'data': {
            'required': False,
        }
    }

    @classmethod
    def additional_info_for_serialization(cls, dataset: DataSet) -> Dict[str, Any]:
        data = dataset.data.to_dict()  # TODO: I feel like I've had problems with this before
        return {'data': data}

    @classmethod
    def additional_info_for_deserialization(cls, d: Dict) -> Dict[str, Any]:
        df = pd.DataFrame(d['data'])
        return {'data': df}
