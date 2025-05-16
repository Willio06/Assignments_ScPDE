from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from geometry import BPoint, Point


class FE(Protocol):
    @property
    def degree(self) -> int: ...
    def __len__(self) -> int: ...

    def value(self, i: int, b: BPoint) -> float: ...
    def grad(self, i: int, b: BPoint) -> Point: ...
    def node(self, i: int) -> BPoint: ...


def create_fe(degree: int) -> FE:
    if degree == 1:
        return FE1()
    elif degree == 2:
        return FE2()
    else:
        raise ValueError(f'FE of degree {degree} not implemented.')


@dataclass(frozen=True)
class FE1:
    degree: int = field(default=1, init=False, repr=False)
    num_nodes: int = field(default=3, init=False, repr=False)

    def __len__(self) -> int:
        return self.num_nodes

    @staticmethod
    def value(i: int, b: BPoint) -> float:
        return b[i]

    @staticmethod
    def grad(i: int, b: BPoint) -> Point:
        if i == 0:
            return Point(np.array([-1., -1.]))
        elif i == 1:
            return Point(np.array([1., 0.]))
        elif i == 2:
            return Point(np.array([0., 1.]))
        else:
            raise ValueError('Index out of bounds')

    @staticmethod
    def node(i: int) -> BPoint:
        res = BPoint(np.array([0., 0., 0.]))
        res[i] = 1.
        return res


@dataclass(frozen=True)
class FE2:
    degree: int = field(default=2, init=False, repr=False)
    num_nodes: int = field(default=6, init=False, repr=False)

    def __len__(self) -> int:
        return self.num_nodes

    @staticmethod
    def value(i: int, b: BPoint) -> float:
        if i < 3:
            return b[i] * (2 * b[i] - 1)
        if i == 3:
            return 4 * b[0] * b[1]
        if i == 4:
            return 4 * b[0] * b[2]
        if i == 5:
            return 4 * b[1] * b[2]
        raise ValueError('Index out of bounds')

    @staticmethod
    def grad(i: int, b: BPoint) -> Point:
        if i == 0:
            return Point(np.array([-3 + 4 * (b[0] + b[1]),
                                   -3 + 4 * (b[1] + b[0])]))
        elif i == 1:
            return Point(np.array([4*b[0]-1, 0.]))
        elif i == 2:
            return Point(np.array([0., 4*b[1]-1]))
        elif i == 3:
            return Point(np.array([-4 * b[1],
                                   4 - 8*b[1] - 4*b[0]]))
        elif i == 4:
            return Point(np.array([4 * b[1],
                                   4 * b[0]]))
        elif i == 5:
            return Point(np.array([4 - 8*b[0] - 4*b[1],
                                   -4 * b[0]]))
        else:
            raise ValueError('Index out of bounds')

    @staticmethod
    def node(i: int) -> BPoint:
        res = BPoint(np.array([0., 0., 0.]))
        if i < 3:
            res[i] = 1.
        elif i == 3:
            res[0] = res[1] = 0.5
        elif i == 4:
            res[0] = res[2] = 0.5
        elif i == 5:
            res[1] = res[2] = 0.5
        else:
            raise ValueError('Index out of bounds')
        return res
