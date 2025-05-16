from __future__ import annotations
import math
from dataclasses import dataclass
from geometry import BPoint, Point, Triangle
from typing import cast

import numpy as np


@dataclass(frozen=True, slots=True)
class QuadratureData:
    degree: int
    quad_points: list[BPoint]
    weights: list[float]


class TriangleTransform:
    """
    These are the mappings F_T : T_ref -> T and F_T^{-1} from the lecture notes
    """
    def __init__(self, triangle: Triangle) -> None:
        # Jacobian of F_T
        J = np.zeros((2, 2))
        J[0][0] = (triangle[1][0] - triangle[0][0])
        J[0][1] = (triangle[2][0] - triangle[0][0])
        J[1][0] = (triangle[1][1] - triangle[0][1])
        J[1][1] = (triangle[2][1] - triangle[0][1])

        self.triangle = triangle
        self.jacobian = J
        self.inverse_jacobian = np.linalg.inv(J)
        
    def F_T(self, x: Point) -> Point:
        return self.J @ x + self.triangle[0]
    
    def F_T_inverse(self, x: Point) -> Point:
        return self.inverse_jacobian @ (x-self.triangle[0])
    
    def transform_grad(self, grad: Point) -> Point:
        return self.inverse_jacobian.T @ grad  # type: ignore[return-value]


def create_quadrature_data(degree: int) -> QuadratureData:
    V = 0.5  # area of unit triangle
    if degree == 3:
        return QuadratureData(
            degree=3,
            quad_points=cast(list[BPoint], [
                np.array([1/3, 1/3, 1/3]),
                np.array([1., 0., 0.]),
                np.array([0., 1., 0.]),
                np.array([0., 0., 1.]),
                np.array([.5, .5, 0.]),
                np.array([.5, 0., .5]),
                np.array([0., .5, .5]),
            ]),
            weights=[
                27./60.*V,
                 3./60.*V,
                 3./60.*V,
                 3./60.*V,
                 8./60.*V,
                 8./60.*V,
                 8./60.*V,
            ],
        )
    elif degree == 5:
        sq15 = math.sqrt(15)
        a  = (6. - sq15)/21
        b  = (9. + 2*sq15)/21
        c  = (6. + sq15)/21
        d  = (9. - 2*sq15)/21
        wm = (155. - sq15) / 1200. * V
        wp = (155. + sq15) / 1200. * V
        return QuadratureData(
            degree=5,
            quad_points=cast(list[BPoint], [
                np.array([1/3, 1/3, 1/3]),
                np.array([a, a, b]),
                np.array([a, b, a]),
                np.array([b, a, a]),
                np.array([c, c, d]),
                np.array([c, d, c]),
                np.array([d, c, c]),
            ]),
            weights=[
                9./40 * V,
                wm,
                wm,
                wm,
                wp,
                wp,
                wp,
            ],
        )
    else:
        raise ValueError(f'Quadrature of degree {degree} not defined.')
