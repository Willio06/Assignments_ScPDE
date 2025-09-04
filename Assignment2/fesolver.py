import time
from collections.abc import Callable
from pathlib import Path

from dof_handler import DofHandler
from fe import create_fe
from geometry import Point, Triangle
from grid import Grid
from quadrature import TriangleTransform, create_quadrature_data

import scipy  # type: ignore[import-untyped]
import matplotlib as mpl  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import argparse


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

    for k, _ in enumerate(grid.triangles):   
        # Get triangle vertex coordinates (as Points)
        triangle_pts = Triangle(tuple([grid.nodes[dofh.local_to_global_map[(k,i)]] for i in range(3)]))
        transform = TriangleTransform(triangle_pts)
        local_A = np.zeros((len(fe), len(fe)))
        local_b = np.zeros(len(fe))
        for qp, w in zip(qd.quad_points, qd.weights):
            det_J = np.abs(np.linalg.det(transform.jacobian))
            x = transform.F_T(qp)
            for i in range(len(fe)):
                grad_i = transform.transform_grad(fe.grad(i, qp))
                for j in range(len(fe)):
                    grad_j = transform.transform_grad(fe.grad(j, qp))
                    local_A[i, j] += w * det_J * np.dot(grad_i, grad_j)
                local_b[i] += w * det_J * rhs(x) * fe.value(i, qp)
        # print(local_b)
        for i_local in range(len(fe)):
            i_global = dofh.local_to_global(k, i_local)
            b[i_global] += local_b[i_local]
            for j_local in range(len(fe)):
                j_global = dofh.local_to_global(k, j_local)
                A[i_global, j_global] += local_A[i_local, j_local]

    # Apply Dirichlet boundary conditions: u = 0 on ∂Ω
    for i, p in enumerate(dofh.nodes):
        if is_point_on_boundary(p):
            A[i, :] = 0
            A[:, i] = 0
            A[i, i] = 1
            b[i] = 0

    return scipy.sparse.csr_array(A), b


def plot_solution(
        dofh: DofHandler,
        x: np.ndarray,
        *,
        plot_grid: bool = True,
        three_d: bool = True,
        save_fig: bool = True
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    print(three_d)
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


def main(filename: str, threeD= True) -> None:
    print(threeD)
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
    plot_solution(dofh, x, three_d=threeD)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="FEM 3D solver", description=" --mesh default=fine, --plot3D  default=true ")
    parser.add_argument("--mesh",default="fine",type=str, help="choose mesh, options are 'coarse' or 'fine', DEFAULT='fine'")
    parser.add_argument("--plot3D",default=1,type=int, help="plot 3D, 1( true) or 2 (false), else 2D plot, DEFAULT=1")
    args = parser.parse_args()
    print(args.plot3D)
    main('data/'+args.mesh+'_mesh.txt', threeD=bool(args.plot3D))
