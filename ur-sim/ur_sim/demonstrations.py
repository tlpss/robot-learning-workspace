from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Demonstration:
    """
    A very simple data container for collecting first person demonstrations
    """

    # must use the default factory! if you simply create
    # observations = [], this becomes a class attribute and is hence
    # shared amongst all instances..
    observations: List[np.ndarray] = field(default_factory=lambda: [])
    actions: List[np.ndarray] = field(default_factory=lambda: [])


if __name__ == "__main__":
    a = Demonstration()
    b = Demonstration()
    a.observations.append(1)
    print(b.observations)
    print(a.observations)
