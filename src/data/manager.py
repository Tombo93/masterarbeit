from abc import ABC, abstractmethod


class Specification(ABC):
    pass


class DataManager(ABC):
    def __init__(self) -> None:
        self.data_specs = None
        self.data = None

    @abstractmethod
    def load(self):
        """_summary_"""

    @abstractmethod
    def transform(self):
        """_summary_"""

    @abstractmethod
    def store(self):
        """_summary_"""
