from abc import abstractmethod

import skorch
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface
from typing import Any, Dict, List, Tuple, Type, Union


def get_instance_import_path(f) -> Dict:
    return f.__module__ + '.' + type(f).__name__


def get_func_or_class_import_path(f) -> Dict:
    return f.__module__ + '.' + f.__name__


class Architecture(ConfigurableComponent):
    """Abstract base class for an `Architecture` class, which must implement a `fit` method."""

    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def fit(self, x, y):
        pass


class ArchitectureInterface(ComponentInterface):

    registered_flavors = {
        'RandomForestClassifier': RandomForestClassifier,
        'LogisticRegression': LogisticRegression,
        'Pipeline': Pipeline,
    }

    config_schema = {
        'flavor': {
            'type': 'string',
        },
        'config': {
            'type': 'dict',
            'default': dict(),
        },
        'name': {
            'type': 'string',
            'required': False,
        },
        'args': {
            'type': 'dict',
            'required': False,
        },
        'instantiate': {
            'type': 'boolean',
            'default': True,
        }
    }

    @classmethod
    def configure(cls, d: Union[Dict, List], **kwargs) -> Union[ConfigurableComponent, Type[ConfigurableComponent],
                                                                List[ConfigurableComponent]]:
        """
        Instantiate a component from a {'flavor: ..., 'config': {}} dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containing the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        if type(d) is list:
            configured_list = []
            for f in d:
                configured_list.append(cls.configure(f))
            return configured_list
        elif type(d) is dict:
            d = cls.validate_config(d)
            flavor_cls = cls.select_flavor(d['flavor'])
            if d.get('instantiate', True) is False:
                return flavor_cls
            elif ArchitectureInterfaceFactory.recognizes_flavor(flavor_cls):
                interface = ArchitectureInterfaceFactory.select_from_flavor(flavor_cls)
                return interface.configure(d)
            else:
                return flavor_cls(**d['config'])
        else:
            raise ValueError(f"Architecture configuration should be of type dict or list, got {type(d)}")

    @classmethod
    def deserialize(cls, d: Dict) -> Union[ConfigurableComponent, Type[ConfigurableComponent]]:
        """
        Instantiate a component from a {'flavor: ..., 'config': {}} dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containting the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        d = cls.validate_config(d)
        flavor_cls = cls.select_flavor(d['flavor'])
        kwargs = cls.additional_info_for_deserialization(d)

        if d.get('instantiate', True) is False:
            return flavor_cls
        elif ArchitectureInterfaceFactory.recognizes_flavor(flavor_cls):
            interface = ArchitectureInterfaceFactory.select_from_flavor(flavor_cls)
            return interface.deserialize(d)
        else:
            return flavor_cls(**d['config'], **kwargs)

    @classmethod
    def serialize(cls, component) -> Dict:
        interface = ArchitectureInterfaceFactory.select_from_component(component)
        if interface is None:
            return super().serialize(component)
        else:
            return interface.serialize(component)


class SKLearnEstimatorInterface(ArchitectureInterface):

    @classmethod
    def configure(cls, d: Dict, **kwargs) -> BaseEstimator:
        flavor_cls = cls.select_flavor(d['flavor'])
        return flavor_cls(**d['config'])

    @classmethod
    def deserialize(cls, d: Dict) -> BaseEstimator:
        return cls.configure(d)

    @classmethod
    def serialize(cls, estimator) -> Dict:
        params = estimator.get_params()
        params = {k: params[k] for k in params.keys() if type(params[k]) != type}
        d = {
            'flavor': get_instance_import_path(estimator),
            'config': params,
        }
        return d


class PipelineInterface(ArchitectureInterface):

    @classmethod
    def configure(cls, d: Dict, **kwargs) -> Pipeline:
        step_list = []
        if 'steps' not in d['config']:
            raise ValueError(f"Pipeline config must contain entry for `steps`, found {list(d['config'].keys())}")
        for step in d['config']['steps']:
            step_obj = ArchitectureInterface.configure(step)
            step_name = step.get('name', step['flavor'])
            step_tuple = (step_name, step_obj)
            step_list.append(step_tuple)
        pipeline_obj = Pipeline(step_list)
        return pipeline_obj

    @classmethod
    def deserialize(cls, d: Dict) -> Pipeline:
        return cls.configure(d)

    @classmethod
    def serialize(cls, pipeline: Pipeline) -> Dict:
        steps = []
        for step in pipeline.steps:
            serialized_step = cls.serialize_pipeline_step(step)
            steps.append(serialized_step)
        d = {
            'flavor': 'sklearn.pipeline.Pipeline',
            'config': {
                'steps': steps,
            }
        }
        print(d)
        return d

    @classmethod
    def serialize_pipeline_step(cls, step_tuple: Tuple[str, Any]) -> Dict:
        name, estimator = step_tuple
        d = ArchitectureInterface.serialize(estimator)
        d['name'] = name
        return d


class ColumnTransformerInterface(ArchitectureInterface):

    @classmethod
    def configure(cls, d: Dict, **kwargs) -> ColumnTransformer:
        step_list = []
        if 'steps' not in d['config']:
            raise ValueError(f"Pipeline config must contain entry for `steps`, found {list(d['config'].keys())}")
        for step in d['config']['steps']:
            step_obj = ArchitectureInterface.configure(step)
            step_name = step.get('name', step['flavor'])
            step_args = step['args']['columns']
            step_tuple = (step_name, step_obj, step_args)
            step_list.append(step_tuple)
        return ColumnTransformer(step_list)

    @classmethod
    def deserialize(cls, d: Dict) -> ColumnTransformer:
        return cls.configure(d)

    @classmethod
    def serialize(cls, column_transformer: ColumnTransformer) -> Dict:
        steps = []
        for step in column_transformer.transformers:
            serialized_step = cls.serialize_column_transformer_step(step)
            steps.append(serialized_step)
        d = {
            'flavor': 'sklearn.compose.ColumnTransformer',
            'config': {
                'steps': steps
            }
        }
        print(d)
        return d

    @classmethod
    def serialize_column_transformer_step(cls, step_tuple: Tuple[str, Any, List[str]]) -> Dict:
        name, estimator, columns = step_tuple
        d = ArchitectureInterface.serialize(estimator)
        d['name'] = name
        d['args'] = {'columns': columns}
        return d


class FunctionTransformerInterface(ArchitectureInterface):

    @classmethod
    def configure(cls, d: Dict, **kwargs) -> FunctionTransformer:
        if 'function' not in d['args']:
            raise ValueError("FunctionTransformer must contain an entry for 'function' in 'args'.")
        func = ArchitectureInterface.configure(d['args']['function'])
        return FunctionTransformer(func, **d['config'])

    @classmethod
    def deserialize(cls, d: Dict) -> FunctionTransformer:
        return cls.configure(d)

    @classmethod
    def serialize(cls, func_transformer: FunctionTransformer) -> Dict:
        params = func_transformer.get_params()
        del params['func']
        d = {
            'flavor': 'sklearn.preprocessing.FunctionTransformer',
            'args': {
                'function':
                    {
                        'flavor': get_func_or_class_import_path(func_transformer.func),
                        'instantiate': False,
                    },
            },
            'config': params
        }
        return d


class SkorchNeuralNetInterface(ArchitectureInterface):

    @classmethod
    def configure(cls, d: Dict, **kwargs) -> skorch.NeuralNet:
        arg_objects = {}
        for arg_name in d['args']:
            arg_obj = ArchitectureInterface.configure(d['args'][arg_name])
            arg_objects[arg_name] = arg_obj
        return skorch.NeuralNet(**arg_objects, **d['config'])

    @classmethod
    def deserialize(cls, d: Dict) -> skorch.NeuralNet:
        return cls.configure(d)

    @classmethod
    def serialize(cls, net: skorch.NeuralNet) -> Dict:
        params = net.get_params()
        exclude = ['module', 'criterion', 'optimizer', 'iterator_train', 'iterator_valid', 'dataset', 'train_split',
                   'callbacks', '_kwargs']
        filtered_params = {k: params[k] for k in params if '__' not in k and k not in exclude}
        if '_kwargs' in params:
            filtered_params = d = {**filtered_params, **params['_kwargs']}

        d = {
            'flavor': 'skorch.NeuralNet',
            'config': filtered_params,
            'args': {
                'module': {
                    'flavor': get_instance_import_path(net.module),
                    'config': net.module.get_params(),
                },
                'criterion': {
                    'flavor': get_func_or_class_import_path(net.criterion),
                    'instantiate': False,
                },
                'optimizer': {
                    'flavor': get_func_or_class_import_path(net.optimizer),
                    'instantiate': False,
                }
            },
        }
        return d


class ArchitectureInterfaceFactory:
    interfaces = {
        Pipeline: PipelineInterface,
        ColumnTransformer: ColumnTransformerInterface,
        skorch.NeuralNet: SkorchNeuralNetInterface,
        FunctionTransformer: FunctionTransformerInterface,
        BaseEstimator: SKLearnEstimatorInterface,
    }

    @classmethod
    def recognizes_flavor(cls, flavor_cls) -> bool:
        """Determine whether the InterfaceFactory knows which interface to use for a flavor."""
        return flavor_cls in cls.interfaces

    @classmethod
    def select_from_flavor(cls, flavor_cls: Type[Union[Pipeline, ColumnTransformer, skorch.NeuralNet,
                                                       FunctionTransformer]]) -> Union[ArchitectureInterface, None]:
        """Based on a flavor type, determine which interface to use to configure/deserialize it."""
        if flavor_cls in cls.interfaces:
            return cls.interfaces[flavor_cls]
        return None

    @classmethod
    def select_from_component(cls, component: Any):
        """Based on a component, determine which interface to use to serialize it."""
        for flavor in cls.interfaces:
            if isinstance(component, flavor):
                return cls.interfaces[flavor]
        return None
