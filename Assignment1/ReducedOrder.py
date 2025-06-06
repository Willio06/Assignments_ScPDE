import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import time
N = 10**3

class ReducedOrder:
    def __init__(self, mus,MU, boundary=0):
        self.mus = mus # mu_i's
        self.n = len(mus) 
        self.MU = MU # the wanted mu
        self.boundary = boundary # adds zero boundary condition at the end if wanted
        self.__MatrixAssembly()
    def __FEMinit(self, mu, decompose=False, FEMsolve=False):
        """
        Initialize the finite element method matrices based on the given mu.
        Solves if wanted, and returns the matrices A and b."""
        h = 1/N
        B1 = (1/h)*(cp.diag(cp.full((N-1,), 2)) + cp.diag(cp.full((N-2,), -1), 1)+ cp.diag(cp.full((N-2,), -1), -1))
        B2 = (h)*(cp.diag(cp.full((N-1,), 2/3)) + cp.diag(cp.full((N-2,), 1/6), 1)+ cp.diag(cp.full((N-2,), 1/6), -1))
        B3 = cp.diag(cp.full((N-2,), 1/2), 1)+ cp.diag(cp.full((N-2,), -1/2), -1)
        A = mu[0]*B1 + mu[1]*B2 + mu[2]*B3
        b = h*cp.ones(N-1)

        if FEMsolve:
            x = cp.linalg.solve(A, b)
            self.FEMx = x
        if decompose:
            return mu[0]*B1, mu[1]*B2, mu[2]*B3,b
        return A, b
    def __FEMspan(self):
        """
        Create a matrix C that spans the finite element space for the given mu_i's."""
        C = cp.zeros((N-1,self.n))
        for i in range(len(self.mus)):
            mu = self.mus[i]
            A, b = self.__FEMinit(mu, FEMsolve=True) #solve the FEM problem for each mu_i
            x = self.FEMx
            C[:, i] = x
        return C
    def __MatrixAssembly(self):
        """
        Assemble the matrices A, B1, B2, B3 and b for the reduced order model."""
        C = self.__FEMspan()
        self.C = C
        B1,B2,B3,b = self.__FEMinit(self.MU, decompose=True, FEMsolve=True) # note that the last femx is this one and thus for the wanted mu
        B1 = cp.transpose(C)@B1@C
        B2 = cp.transpose(C)@B2@C
        B3 = cp.transpose(C)@B3@C
        b = cp.transpose(C)@b
        A = B1 + B2 + B3
        self.A = A
        self.C = C
        self.b = b
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3
    def solve(self):
        x = cp.linalg.solve(self.A, self.b)
        self.x = self.C@x
        return self.x
    def addBoundary(self, x: cp.ndarray):
        """Add a boundary condition to the solution."""
        # will be useless because the boundary condition is not approximated and thus will always have zero error value for any approximation with
        # added boundary condition, thus it is equivalent to just leaving it out
        x = cp.append(x, self.boundary)
        x = cp.concatenate((cp.array([self.boundary]),x))
        return x
        

class P:
    """ class to represent and generate points in the parameter space P and P_hat
        alternative=True means that the points are generated in the alternative way
    """
    def __init__(self, n=None, alternative=False): 
        self.n = int(np.cbrt(n))
        self.x_range = (1e-1, 1.0)
        self.y_range = (0.0, 1.0)
        self.z_range = (0.0, 1.0)
        self.x2_range = (1e-2, 1.0)
        self.alternative = alternative

    def __iter__(self):
        """Iterate over the points in the parameter space."""
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
        """Generate random points in the parameter space."""
        x = np.random.uniform(*self.x_range, size=size)
        y = np.random.uniform(*self.y_range, size=size)
        z = np.random.uniform(*self.z_range, size=size)
        return np.stack((x, y, z), axis=-1)

def Vrom(p,greedy=True, n=500):
    """ construction of Vrom by picking mu_i in random or greedy way"""
    if(not greedy):
        return P(10).random(size=n)
    else:
        """ this is just the initialization of  the greedy algo. To add greedly points, use the error function with greed=True"""
        Mnorm = cp.array([]) # array to store the H2 norms of the FEM solutions
        for mu in p:
            femmer = ReducedOrder([mu], mu, boundary=0)# take for mus just mu, calculations will go faster with less mu, is not relevant here anyway just the FEM solution is needed
            x = femmer.FEMx
            Mnorm = cp.append(Mnorm,H2_norm_approx(x, h=1/N)) # compute the H2 norm of the FEM solution
        selected = [p[cp.argmax(Mnorm).get()]] # start with the point with the highest H2 norm
        return selected

def mixed_difference(x,h=1):
    """ use finite difference method to compute derivative of array for H1 norm approximation"""
    diffs = cp.empty_like(x)
    diffs[:-1] = x[1:] - x[:-1]  # forward differences
    diffs[-1] = x[-1] - x[-2]    # backward difference for last element
    return diffs/h
def H2_norm_approx(x,h=1):
    """
    Use trapezoidal rule to approximate the H2 norm of a 1D array.
    """
    x = cp.abs(x) # note absolute value
    diffs = mixed_difference(x, h)
    # Use trapezoidal rule for numerical integration
    integral1 = cp.trapz(diffs**2, dx=h)
    integral2 = cp.trapz(x**2, dx=h)
    return cp.asnumpy(cp.sqrt(integral1 + integral2))
def error(mus,P, greed=False):
    listy=cp.array([]) # list to store the errors
    for mu in P: # iterate over all mu's that are not in mus
        # print(" Computing error for mu:", mu, " at index", len(listy))
        mu = mu.tolist()
        rb = ReducedOrder(mus,mu, boundary=0)
        x= rb.solve()
        xfem = rb.FEMx
        listy = cp.append(listy,H2_norm_approx(x-xfem, h=1/N))
    if greed:
        diff = [mu for mu in P if not np.any(np.all(mus == mu, axis=1))]#set difference between P and mus
        mus.append(diff[cp.argmax(listy[np.all(~np.isin(P,mus), axis = 1)]).get()]) # take greedily max without duplicates, so in set difference
        return cp.max(listy), listy, mus
    return cp.max(listy), listy



def ex3():
    error1=[]
    t = time.time()
    p =  P(10).random(size=500)
    for n in range(2,51):
        print("Computing non-greedy ROM for n =", n, " time elapsed:", time.time() - t)
        t = time.time()
        Vrom1 = Vrom(p,greedy=False, n=n)
        error1.append(error(Vrom1,p,greed=False)[0].get())
    error2=[]
    
    Vrom1 = Vrom(p,greedy=True, n=1)
    _,_, Vrom1 = error(Vrom1, p, greed=True) # initialize with the first point
    for n in range(2,51):
        print("Computing greedy ROM for n =", n, " time elapsed:", time.time() - t)
        t = time.time()
        err,_, Vrom1 = error(Vrom1, p, greed=True) # points are added greedily here already
        error2.append(err.get())
    plt.plot(range(2,51), error1, label="Non-Greedy ROM")
    plt.plot(range(2,51), error2, label="Greedy ROM")
    plt.xlabel("Number of basis functions")
    print(error1)
    print(error2)
    plt.ylabel("Argmax H2 norm error")
    plt.title("Comparison of Greedy and Non-Greedy ROM")
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
ex3()