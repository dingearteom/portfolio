import abc


class ModelInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self):
        pass
