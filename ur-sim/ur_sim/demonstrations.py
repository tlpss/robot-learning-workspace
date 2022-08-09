import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import imageio
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


def save_visual_demonstrations(demonstrations: List[Demonstration], path):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    for i, demo in enumerate(demonstrations):
        demonstration_path = path / f"{i}"
        demonstration_path.mkdir()

        for j, obs in enumerate(demo.observations):
            imageio.imwrite(demonstration_path / f"{j}.jpg", obs)

        with open(str(demonstration_path / "actions.pkl"), "wb") as file:
            pickle.dump(demo.actions, file)


if __name__ == "__main__":
    a = Demonstration()
    b = Demonstration()
    a.observations.append(1)
    print(b.observations)
    print(a.observations)
