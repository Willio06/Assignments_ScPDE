from dataclasses import dataclass
from typing import NewType

import numpy as np


Point = NewType('Point', np.ndarray)

# Barycentric coordinate
BPoint = NewType('BPoint', np.ndarray)


@dataclass
class Triangle:
    nodes: tuple[Point, Point, Point]

    def __getitem__(self, i: int) -> Point:
        return self.nodes[i]
