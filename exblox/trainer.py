from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple
from .architecture import Architecture, ArchitectureInterface
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface
from .tuner import Tuner, TunerInterface
from .predictor import Predictor


class Trainer(ConfigurableComponent):

    def __init__(self, config: Dict, architecture: Architecture, tuner: Tuner = None):
        if not isinstance(config, dict):
            if isinstance(architecture, dict) or isinstance(tuner, dict):
                raise ValueError('It looks like the order of the arguments to `Trainer` is swapped. Please use'
                                 ' `Trainer(config, architecture, tuner)`.')
            raise ValueError('The first argument to Trainer must be a dictionary.')
        super().__init__(config)
        self.architecture = architecture
        self.tuner = tuner

    def fit(self, x, y) -> Tuple[Predictor, dict]:
        """
        Train the `Architecture`, yielding a trained `Predictor` object and a `training_metadata` dictionary.

        If `Trainer` contains a `Tuner`, use it to fit the `Architecture`. Otherwise, `fit` the `Architecture` directly.

        Args:
            x: The training set.
            y: The labels for the training set.

        Returns: a trained `Predictor` object and a `training_metadata` dictionary.

        """
        training_metadata = {}
        architecture = self.get_architecture()
        if self.tuner:
            trained_model, training_metadata = self.tuner.fit(architecture, x, y)
        else:
            trained_model = architecture.fit(x, y)
        predictor = Predictor(trained_model)
        return predictor, training_metadata

    def get_architecture(self) -> Architecture:
        """
        Instantiate a new copy of an untrained architecture.

         Using this method rather than referencing `self.architecture` ensures that a single `Trainer` object can be
          used to train and yield multiple architectures, without overwriting the origianl (by reference).

        Returns: A new architecture object

        """
        fresh_architecture = deepcopy(self.architecture)
        return fresh_architecture


class SkorchTrainer(Trainer):

    def fit(self, x, y) -> Tuple[Predictor, Dict]:
        training_metadata = {}
        architecture = self.get_architecture()
        y = self.transform_y(y)

        if self.tuner:
            trained_model, training_metadata = self.tuner.fit(architecture, x, y)
        else:
            trained_model = architecture.fit(x, y)
        predictor = Predictor(trained_model)
        return predictor, training_metadata

    @staticmethod
    def transform_y(y):
        y = LabelEncoder().fit_transform(y)
        y = y.astype('float32')
        y = y.reshape((len(y), 1))
        return y


class TrainerInterface(ComponentInterface):

    registered_flavors = {
        'Trainer': Trainer,
        'SkorchTrainer': SkorchTrainer,
    }

    @classmethod
    def deserialize(cls, d: Dict[str, Dict]) -> Trainer:
        """
        Instantiate a Trainer from dictionary containing keys ['Trainer', 'Architecture', 'Tuner'].

        Args:
            d: A dictionary with keys ['Trainer', 'Architecture', 'Tuner'], each containing configuration dictionaries.
             The configuration dictionaries contain the key 'flavor' describing the class name of the component to be
             instantiated, and key 'config' containing the object's config dictionary. The configuration dictionaries
             may also contain other keys, which must be added to the object by the subclass-ed deserialize method.

        Returns:
            A deserialized Trainer object.
        """
        trainer_config = d['Trainer']
        trainer_config = cls.validate_serialization_config(trainer_config)
        flavor_cls = cls.select_flavor(trainer_config['flavor'])

        architecture = ArchitectureInterface.deserialize(d['Architecture'])
        tuner = TunerInterface.deserialize(d['Tuner']) if 'Tuner' in d else None

        flavor_instance = flavor_cls(config=trainer_config['config'], architecture=architecture, tuner=tuner)
        return flavor_instance
