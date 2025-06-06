import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import time
N = 10

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
    def addBoundary(self, x):
        """Add a boundary condition to the solution."""
        # will be useless because the boundary condition is not approximated and thus will always have zero error value for any approximation with
        # added boundary condition, thus it is equivalent to just leaving it out
        x = cp.append(x, self.boundary)
        x = cp.concatenate((cp.array([self.boundary]),x))
        return x
    def plotter(self, x, y=None, title=None, xlabel=None, ylabel=None, logscale=False, labelx=None, labely=None):
        size_x  = np.shape(x)[0]

        x_step = np.arange(0, 1, 1/size_x)

        plt.plot(x_step, x, drawstyle='default', color='blue', linewidth=2,label=labelx)
        plt.xlim(0,1)
        # plt.ylim(0, max(self.x) * 1.1)
       
        if y is not None:
            size_y  = np.shape(y)[0]
            y_step = np.arange(0,1, 1/size_y)
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
    """ class to represent and generate points in the parameter space P and P_hat
        alternative=True means that the points are generated in the alternative way
    """
    def __init__(self, n=None, alternative=False): 
        if n is not None:
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
        if self.alternative:
            x= np.random.uniform(*self.x2_range, size=size)
        else:
            x = np.random.uniform(*self.x_range, size=size)
        y = np.random.uniform(*self.y_range, size=size)
        z = np.random.uniform(*self.z_range, size=size)
        return np.stack((x, y, z), axis=-1)

def VromGreedy(p,initial=False, VromPrev=None):
    if(initial): #initial case
        Mnorm = cp.array([]) # array to store the H2 norms of the FEM solutions
        for mu in p:
            femmer = ReducedOrder([mu], mu, boundary=0)# take for mus just mu, calculations will go faster with less mu, is not relevant here anyway just the FEM solution is needed
            x = femmer.FEMx
            Mnorm = cp.append(Mnorm,H2_norm_approx(x, h=1/N)) # compute the H2 norm of the FEM solution
        selected = [p[cp.argmax(Mnorm).get()]] # start with the point with the highest H2 norm
        return selected
    elif VromPrev is not None:
        Mnorm = cp.array([])
        setdiff = [mu for mu in p if not np.any(np.all(VromPrev == mu, axis=1))]
        for mu in setdiff:
            femmer = ReducedOrder(VromPrev, mu, boundary=0)
            xfem = femmer.FEMx
            xrb = femmer.solve()
            Mnorm = cp.append(Mnorm, H2_norm_approx(xfem - xrb, h=1/N))  # compute the H2 norm of the difference
        selected = [setdiff[cp.argmax(Mnorm).get()]]  # select the point with the highest H2 norm
        return np.append(VromPrev, selected, axis=0)



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


def error(mus, testP):
    listy=cp.array([]) # list to store the errors
    for mu in testP: # iterate over all mu's that are not in mus
        # print(" Computing error for mu:", mu, " at index", len(listy))
        mu = mu.tolist()
        rb = ReducedOrder(mus,mu, boundary=0)
        x= rb.solve()
        xfem = rb.FEMx
        listy = cp.append(listy,H2_norm_approx(x-xfem, h=1/N))
    return cp.max(listy), listy

def ex3():
    alternative = False
    testP = P(alternative=alternative).random(size=500) # generate random points in the parameter space for testing
    error1=[]
    t = time.time()
    Vrom1 = P(alternative=alternative).random(size=1) # generate random point in the parameter space
    for n in range(2,51):
        print("Computing non-greedy ROM for n =", n, " time elapsed:", time.time() - t)
        t = time.time()
        Vrom1  = np.append(Vrom1, P(alternative=alternative).random(size=1), axis=0) # add randomly
        error1.append(error(Vrom1, testP)[0].get())
    
    
    error2=[] 
    pTrain = P(alternative=alternative).random(size=500) # generate random points in the parameter space for training
    Vrom1 = VromGreedy(pTrain,initial=True) # initialize the greedy ROM with the first point
    for n in range(2,51):
        print("Computing greedy ROM for n =", n, " time elapsed:", time.time() - t)
        t = time.time()
        Vrom1 = VromGreedy(pTrain, VromPrev=Vrom1) # greedily add to Vrom
        error2.append(error(Vrom1, testP)[0].get())
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
def test():
    PP = P(10).random(size=2)
    rb = ReducedOrder(PP, [0.1,0,1], boundary=0)
    x = rb.solve()
    x = rb.addBoundary(x)  # add boundary condition if needed
    xfem = rb.FEMx
    xfem = rb.addBoundary(xfem)  # add boundary condition if needed
    rb.plotter(cp.asnumpy(x), cp.asnumpy(xfem), title="Reduced Order Model vs Finite Element Method", xlabel="x", ylabel="u(x)", labelx="ROM", labely="FEM")
    print("H2 norm error:", H2_norm_approx(x-xfem, h=1/N))
    print("error: ", error(PP, P(10).random(size=500), greed=False)[0].get())
    plt.show()