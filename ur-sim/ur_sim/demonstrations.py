import dataclasses


class Demonstration(dataclasses.dataclass):
    """
    A very simple data container for collecting first person demonstrations
    """
    observations = []
    actions = []
