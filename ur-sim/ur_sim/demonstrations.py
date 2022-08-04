import dataclasses

@dataclasses.dataclass
class Demonstration:
    """
    A very simple data container for collecting first person demonstrations
    """
    observations = []
    actions = []
