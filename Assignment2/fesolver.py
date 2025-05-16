import time
from collections.abc import Callable
from pathlib import Path

from dof_handler import DofHandler
from fe import create_fe
from geometry import Point
from grid import Grid
from quadrature import TriangleTransform, create_quadrature_data

import scipy  # type: ignore[import-untyped]
import matplotlib as mpl  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np


def galerkin_discretization(
        dofh: DofHandler,
        rhs: Callable[[Point], float],
        is_point_on_boundary: Callable[[Point], bool],
) -> tuple[scipy.sparse.csr_array, np.ndarray]:
    A = np.zeros((len(dofh), len(dofh)))
    b = np.zeros(len(dofh))
    qd = create_quadrature_data(5)
    fe = dofh.fe
    grid = dofh.grid

    # TODO: Fill stiffness matrix A and rhs vector b.

    # TODO: Set zero boundary values.

    return scipy.sparse.csr_array(A), b


def plot_solution(
        dofh: DofHandler,
        x: np.ndarray,
        *,
        plot_grid: bool = True,
        three_d: bool = True,
        save_fig: bool = True
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    Path('solution').mkdir(exist_ok=True)
    xs = np.empty(len(dofh))
    ys = np.empty(len(dofh))
    for i, p in enumerate(dofh.nodes):
        xs[i] = p[0]
        ys[i] = p[1]
    if not three_d:
        fig, ax = plt.subplots()
        pos = ax.tripcolor(xs, ys, x, shading='gouraud')
        fig.colorbar(pos, ax=ax)
        if plot_grid:
            xs = np.empty(len(dofh.grid.nodes))
            ys = np.empty(len(dofh.grid.nodes))
            for i, p in enumerate(dofh.grid.nodes):
                xs[i] = p[0]
                ys[i] = p[1]
            ax.triplot(xs, ys, dofh.grid.triangles)
        if save_fig:
            plt.savefig('solution/solution-2d.png')
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        pos = ax.plot_trisurf(xs, ys, x, cmap=plt.cm.viridis)
        fig.colorbar(pos, ax=ax)
        if save_fig:
            plt.savefig('solution/solution-3d.png')
    return fig, ax


def main(filename: str) -> None:
    t = time.monotonic()
    print(f'Reading {filename}... ', end='', flush=True)
    grid = Grid.read_from_file(filename)
    print(f'took {time.monotonic() - t:.4f}s')
    t = time.monotonic()
    print('Creating DoF handler... ', end='', flush=True)
    fe = create_fe(1)
    dofh = DofHandler(grid, fe)
    print(f'took {time.monotonic() - t:.3f}s')

    t = time.monotonic()
    print('Filling stiffness matrix... ', end='', flush=True)
    A, b = galerkin_discretization(
            dofh,
            lambda x: 1.,
            lambda x: np.linalg.norm(x, ord=np.inf) == 1)
    print(f'took {time.monotonic() - t:.3f}s')

    t = time.monotonic()
    print('Solving linear system... ', end='', flush=True)
    x = scipy.sparse.linalg.spsolve(A, b)
    print(f'took {time.monotonic() - t:.3f}s')
    plot_solution(dofh, x)
    plt.show()


if __name__ == '__main__':
    main('data/coarse_mesh.txt')
