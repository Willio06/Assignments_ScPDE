from fe import FE
from grid import Grid

from geometry import Point

import numpy as np


class DofHandler:
    def __init__(self, grid: Grid, fe: FE):
        self.fe = fe
        self.grid = grid
        self.local_to_global_map = {}
        
        for i, triangle in enumerate(grid.triangles):
            for k in range(len(triangle)):
                self.local_to_global_map[(i, k)] = triangle[k]
        
        self.nodes = grid.nodes
    def __len__(self) -> int:
        return len(self.nodes)

    def local_to_global(self, i_cell: int, i_fe: int) -> int:
        return self.local_to_global_map[i_cell, i_fe]

    def global_node(self, i: int) -> Point:
        return self.nodes[i]
