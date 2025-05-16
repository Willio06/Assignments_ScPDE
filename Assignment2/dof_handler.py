from fe import FE
from grid import Grid

from geometry import Point

import numpy as np


class DofHandler:
    def __init__(self, grid: Grid, fe: FE):
        # TODO: Create a dictionary, called self.local_to_global_map, that maps pairs of triangle index k and local Lagrange node index i to the index of the corresponding global node.
        self.fe = fe
        self.grid = grid
        # TODO: Store global DoF coordinates in self.nodes.

    def __len__(self) -> int:
        return len(self.nodes)

    def local_to_global(self, i_cell: int, i_fe: int) -> int:
        return self.local_to_global_map[i_cell, i_fe]

    def global_node(self, i: int) -> Point:
        return self.nodes[i]
