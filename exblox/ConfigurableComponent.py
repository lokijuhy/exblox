from abc import ABC
from cerberus import Validator
import importlib
from typing import Any, Dict, Type


class ConfigurableComponent(ABC):

    def __init__(self, config: Dict = None):
        self.config = config
        if self.config is None:
            self.config = {}


class ComponentInterface:

    registered_flavors = {}

    config_schema = {
        'flavor': {
            'type': 'string',
        },
        'config': {
            'type': 'dict',
            'default': dict(),
        }
    }
    serialization_schema = {}

    @classmethod
    def serialize(cls, component) -> Dict:
        d = {
            'flavor': type(component).__name__,
            'config': component.config,
        }
        more_info = cls.additional_info_for_serialization(component)
        d = {**d, **more_info}
        cls.validate_serialization_config(d)
        return d

    @classmethod
    def additional_info_for_serialization(cls, component: ConfigurableComponent) -> Dict[str, Any]:
        """
        (Optional) Build a dictionary of additional data (beyond the config) that needs to be saved in the
         serialization dictionary. The key value pairs will be added to the object's serialization dictionary.

        Args:
            component: The component from which to serialize additional information.

        Returns: A dictionary of additional information to be included in the serialization dictionary.
        """
        return {}

    @classmethod
    def configure(cls, d: Dict, **kwargs) -> ConfigurableComponent:
        """
        Instantiate a component from a dictionary and other objects if necessary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be instantiated, and
             key 'config' containing the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        d = cls.validate_config(d)
        flavor_cls = cls.select_flavor(d['flavor'])
        flavor_instance = flavor_cls(config=d['config'], **kwargs)
        return flavor_instance

    @classmethod
    def deserialize(cls, d: Dict) -> ConfigurableComponent:
        """
        Instantiate a component from a dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be instantiated, and
             key 'config' containing the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        d = cls.validate_serialization_config(d)
        flavor_cls = cls.select_flavor(d['flavor'])
        kwargs = cls.additional_info_for_deserialization(d)
        flavor_instance = flavor_cls(config=d['config'], **kwargs)
        return flavor_instance

    @classmethod
    def additional_info_for_deserialization(cls, d: Dict) -> Dict[str, Any]:
        """
        (Optional) Build a dictionary of additional data (beyond the config) that needs to be extracted from the
         serialization dictionary in order to initialize the component. The key value pairs will be passed to the
          component initialization as kwargs.

        Args:
            d: The serialization dictionary from which to extract additional information for initialization.

        Returns: A dictionary of additional information to be included passed to the component's init.
        """
        return {}

    @classmethod
    def select_flavor(cls, flavor: str) -> Type[Any]:
        if flavor in cls.registered_flavors:
            return cls.registered_flavors[flavor]
        elif '.' in flavor:
            return cls.load_library_flavor(flavor)
        else:
            raise ValueError(f"{cls.__name__} '{flavor}' is not among the registered flavors and does not appear to be"
                             f" a reference to a library.")

    @classmethod
    def load_library_flavor(cls, flavor):
        try:
            library_path = flavor.split('.')
            # TODO: this is a total hack!
            library_path_without_underscores = [part for part in library_path[0:-1] if not part.startswith('_')]
            library = '.'.join(library_path_without_underscores)
            flavor_class = library_path[-1]
            model = getattr(importlib.import_module(library), flavor_class)
            return model
        except ImportError:
            raise ValueError(f"{cls.__name__} '{flavor}' appears to be a reference to a library, but was not"
                             f" importable.")

    @classmethod
    def validate_config(cls, d: Dict) -> Dict:
        v = Validator(cls.config_schema)
        d_norm = v.normalized(d)
        if not v.validate(d_norm):
            raise ValueError(f"{cls.__name__} encountered config validation errors: {v.errors}")
        return d_norm

    @classmethod
    def validate_serialization_config(cls, d: Dict) -> Dict:
        # compile config and serialization schemas
        schema = {}
        for key in cls.config_schema:
            schema[key] = cls.config_schema[key]
        for key in cls.serialization_schema:
            schema[key] = cls.serialization_schema[key]

        v = Validator(schema)
        d_norm = v.normalized(d)
        if not v.validate(d_norm):
            raise ValueError(f"{cls.__name__} encountered config validation errors: {v.errors}")
        return d_norm
