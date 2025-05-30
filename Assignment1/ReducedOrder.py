from FiniteElements import FiniteElements_1D
import numpy as np
import matplotlib.pyplot as plt
import warnings

N = 10**3

class ReducedOrder:
    def __init__(self, mus,MU, boundary=0, added_boundary=False):
        self.mus = mus
        self.n = len(mus)
        self.MU = MU
        self.boundary = boundary
        self.added_boundary = added_boundary
        self.__MatrixAssembly()
    def __FEMinit(self, mu, decompose=False, FEMsolve=False):
        h = 1/N
        B1 = (1/h)*(np.diag((N-1)*[2]) + np.diag((N-2)*[-1], 1)+ np.diag((N-2)*[-1], -1))
        B2 = (h)*(np.diag((N-1)*[2/3]) + np.diag((N-2)*[1/6], 1)+ np.diag((N-2)*[1/6], -1))
        B3 = np.diag((N-2)*[1/2], 1)+ np.diag((N-2)*[-1/2], -1)
        A = mu[0]*B1 + mu[1]*B2 + mu[2]*B3
        b = h*np.ones(N-1)
        if decompose:
            return mu[0]*B1, mu[1]*B2, mu[2]*B3,b
        x = np.linalg.solve(A, b)
        if FEMsolve:
            if not self.added_boundary:
                x = np.append(x, self.boundary)
                x = np.insert(x, 0,self.boundary)
            self.FEMx = x
        return A, b
    def __FEMspan(self):
        C = np.zeros((N-1,self.n))
        for mu in self.mus:
            A, b = self.__FEMinit(mu)
            fem = FiniteElements_1D(0, 1, A, b, boundary=0, added_boundary=True)
            x = fem.solve()
            C[:, self.mus.index(mu)] = x
        return C
    def __MatrixAssembly(self):
        C = self.__FEMspan()
        B1,B2,B3,b = self.__FEMinit(self.MU, decompose=True, FEMsolve=True)
        B1 = np.transpose(C)*B1*C
        B2 = np.transpose(C)*B2*C
        B3 = np.transpose(C)*B3*C
        b = np.transpose(C)*b
        A = B1 + B2 + B3
        self.A = A
        self.C = C
        self.b = b
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3
    def solve(self):
        x = np.linalg.solve(self.A, self.b)
        if not self.added_boundary:
            x = np.append(x, self.boundary)
            x = np.insert(x, 0,self.boundary)
        self.x = x
        return x
    def plotter(self, x, y=None, title=None, xlabel=None, ylabel=None, logscale=False, labelx=None, labely=None):
        size_x  = np.shape(x)[0]

        x_step = np.arange(self.start, self.end, (self.end-self.start)/size_x)

        plt.plot(x_step, x, drawstyle='default', color='blue', linewidth=2,label=labelx)
        plt.xlim(self.start, self.end)
        # plt.ylim(0, max(self.x) * 1.1)
       
        if y is not None:
            size_y  = np.shape(y)[0]
            y_step = np.arange(self.start, self.end, (self.end-self.start)/size_y)
            plt.plot(y_step, y, drawstyle='default', color='red', linewidth=2,label=labelx)
            plt.legend()


        plt.xlabel(xlabel if xlabel else 'X axis')
        plt.ylabel(ylabel if ylabel else 'Y axis')
        if logscale:
            plt.yscale('log')
        else:
            plt.yscale('linear')
        plt.title(title if title else 'Plot title')
        plt.grid(True, linestyle='--', alpha=0.5)
class P:
    def __init__(self, n=None, alternative=False):
        self.n = int(np.cbrt(n))
        self.x_range = (1e-1, 1.0)
        self.y_range = (0.0, 1.0)
        self.z_range = (0.0, 1.0)
        self.x2_range = (1e-2, 1.0)
        self.alternative = alternative

    def __iter__(self):
        xs = np.linspace(*self.x_range, self.n)
        ys = np.linspace(*self.y_range, self.n)
        zs = np.linspace(*self.z_range, self.n)
        x2s = np.linspace(*self.x2_range, self.n)
        if self.alternative:
            xs, ys, zs = np.meshgrid(xs, ys, zs, indexing='ij')
            grid = np.stack((xs.ravel(), ys.ravel(), zs.ravel()), axis=-1)
            for point in grid:
                yield point
        else:
            x, y, z = np.meshgrid(xs, ys, zs, indexing='ij')
            grid = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=-1)
            for point in grid:
                yield point
    def random(self, size=1):
        x = np.random.uniform(*self.x_range, size=size)
        y = np.random.uniform(*self.y_range, size=size)
        z = np.random.uniform(*self.z_range, size=size)
        return np.stack((x, y, z), axis=-1)

def Vrom(greedy=True, n=500):
    if(not greedy):
        return P(10).random(size=n)
    else:
        p = P(10).random(size=500)
        selected = []
        for i in range(n):
            norms = []
            for k in range(500):
                if i==0:
                    rb = ReducedOrder(p[k], p[k], boundary=0, added_boundary=False)
                else:
                    rb = ReducedOrder(selected, p[k], boundary=0, added_boundary=False)
                x = rb.solve()
                xfem = rb.FEMx
                norms.append(H2_norm_approx(x-xfem, h=1/N))
            i = np.argmax(norms)
            selected.append(p[i])
        return selected
def mixed_difference(x,h=1):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array")

    n = len(x)
    if n < 2:
        raise ValueError("Array must contain at least two elements")

    diffs = np.empty_like(x)
    diffs[:-1] = x[1:] - x[:-1]  # forward differences
    diffs[-1] = x[-1] - x[-2]    # backward difference for last element
    return diffs/h
def H2_norm_approx(x,h=1):
    """
    Use trapezoidal rule to approximate the H2 norm of a 1D array.
    """
    x = np.abs(np.asarray(x))
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array")

    n = len(x)
    if n < 2:
        raise ValueError("Array must contain at least two elements")

    diffs = mixed_difference(x, h)
    # Use trapezoidal rule for numerical integration
    integral1 = np.trapz(diffs**2, dx=h)
    integral2 = np.trapz(x**2, dx=h)
    return np.sqrt(integral1 + integral2)
def error(mus):
    listy=[]
    for mu in P(10).random(size=500):
        mu = mu.tolist()
        rb = ReducedOrder(mus,mu, boundary=0, added_boundary=True)
        x= rb.solve()
        xfem = rb.FEMx
        listy.apppend(H2_norm_approx(x-xfem, h=1/N))
    return np.max(listy), listy



def ex3():
    error1=[]
    for n in range(2,51):
        Vrom1 = Vrom(greedy=False, n=n)
        error1.append(error(Vrom1)[0])
    error2=[]
    for n in range(2,51):
        Vrom1 = Vrom(greedy=True, n=n)
        error2.append(error(Vrom1)[0])
    plt.plot(range(2,51), error1, label="Non-Greedy ROM")
    plt.plot(range(2,51), error2, label="Greedy ROM")
    plt.xlabel("Number of basis functions")
    plt.ylabel("Argmax H2 norm error")
    plt.title("Comparison of Greedy and Non-Greedy ROM")
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
ex3()