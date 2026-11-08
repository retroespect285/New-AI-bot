from .base import Broker, BrokerError
from .sim import SimBroker

__all__ = ["Broker", "BrokerError", "SimBroker", "MT5Broker"]


def __getattr__(name):
    if name == "MT5Broker":  # imported lazily: MetaTrader5 is Windows-only
        from .mt5 import MT5Broker

        return MT5Broker
    raise AttributeError(name)
