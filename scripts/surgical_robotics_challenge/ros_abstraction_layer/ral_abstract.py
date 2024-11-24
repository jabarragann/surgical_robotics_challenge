from abc import ABC, abstractmethod


class RAL_Abstract(ABC):

    @abstractmethod
    def __init__(self, node):
        pass

    @abstractmethod
    def identify(self) -> str:
        """
        Return either ROS1 or ROS2
        """
        pass

    @abstractmethod
    def is_shutdown(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass

    @abstractmethod
    def create_rate(self, rate):
        pass

    @abstractmethod
    def now(self):
        pass
