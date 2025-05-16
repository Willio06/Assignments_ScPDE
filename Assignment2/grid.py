from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from geometry import Point, Triangle

@dataclass
class Grid:
    nodes: list[Point]
    triangles: list[tuple[int, int, int]]

    @property
    def num_vertices(self) -> int:
        return len(self.nodes)

    @property
    def num_triangles(self) -> int:
        return len(self.triangles)

    def __iter__(self) -> Iterator[Triangle]:
        """Iterate over the triangles in the grid.

        The triangles are iterated over in the order in which they have
        been read from the input file.
        """
        for t in self.triangles:
            nodes = cast(tuple[Point, Point, Point],
                         tuple(self.nodes[i] for i in t))
            yield Triangle(nodes=nodes)

    @classmethod
    def read_from_file(cls, file: Path | str) -> Grid:
        with open(file) as f:
            def read_line_skipping_comments() -> str:
                line = f.readline()
                while line.lstrip().startswith('#'):
                    line = f.readline()
                return line

            num_vertices = int(read_line_skipping_comments().strip())
            num_triangles = int(read_line_skipping_comments().strip())
            line = read_line_skipping_comments()
            if line != 'Vertices\n':
                raise RuntimeError(
                        'Vertices list must begin with keyword "Vertices", '
                        f'found {line!r} instead.')
            nodes = cast(list[Point],
                         [np.array([float(p)
                                    for p in f.readline().strip().split()])
                          for _ in range(num_vertices)])
            line = read_line_skipping_comments()
            if line != 'Triangles\n':
                raise RuntimeError(
                        'Triangles list must begin with keyword "Triangles", '
                        f'found {line!r} instead.')
            # We count nodes from 0 whereas the input file counts from 1.
            triangles = cast(list[tuple[int, int, int]],
                             [tuple(int(p)-1
                                    for p in f.readline().strip().split())
                              for _ in range(num_triangles)])
            line = read_line_skipping_comments()
            if line != 'EndTriangles\n':
                raise RuntimeError(
                        'Triangles list must end with keyword "EndTriangles", '
                        f'found {line!r} instead.')
        return cls(nodes=nodes, triangles=triangles)


if __name__ == '__main__':
    Grid.read_from_file(Path('data/coarse_mesh.txt'))
