import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import time
N = 10**3 #FEM dimension
FEMdict={} #speed up calculations by saving FEM solutions globally
class ReducedOrder:
    global FEMdict
    def __init__(self, mus=None,MU=None, boundary=0):
        if mus is not None:
            self.mus = mus # mu_i's
            self.n = len(mus) 
        if MU is not None:
            self.MU = MU # the wanted mu
        self.boundary = boundary # adds zero boundary condition at the end if wanted
        if mus is not None and MU is not None:
            self.__MatrixAssembly()
    def __FEMinit(self, mu, decompose=False):
        """
        Initialize the finite element method matrices based on the given mu.
        Solves if wanted, and returns the matrices A and b."""
        h = 1/N
        B1 = (1/h)*(cp.diag(cp.full((N-1,), 2)) + cp.diag(cp.full((N-2,), -1), 1)+ cp.diag(cp.full((N-2,), -1), -1))
        B2 = (h)*(cp.diag(cp.full((N-1,), 2/3)) + cp.diag(cp.full((N-2,), 1/6), 1)+ cp.diag(cp.full((N-2,), 1/6), -1))
        B3 = cp.diag(cp.full((N-2,), 1/2), 1)+ cp.diag(cp.full((N-2,), -1/2), -1)
        A = mu[0]*B1 + mu[1]*B2 + mu[2]*B3
        b = h*cp.ones(N-1)
        if decompose:
            return mu[0]*B1, mu[1]*B2, mu[2]*B3,b
        return A, b
    def FEMsolve(self,mu=None):
        if mu is None: mu = self.MU
        if str(mu) in FEMdict.keys():
            return FEMdict[str(mu)]
        else:
            A,b = self.__FEMinit(mu)
            self.FEMx = cp.linalg.solve(A,b)
            FEMdict[str(mu)] = self.FEMx
        return self.FEMx
    def __FEMspan(self):
        """
        Create a matrix C that spans the finite element space for the given mu_i's."""
        C = cp.zeros((N-1,self.n))
        for i in range(len(self.mus)):
            mu = self.mus[i]
            C[:, i] = self.FEMsolve(mu)
        return C
    def __MatrixAssembly(self):
        """
        Assemble the matrices A, B1, B2, B3 and b for the reduced order model."""
        C = self.__FEMspan()
        self.C = C
        B1,B2,B3,b = self.__FEMinit(self.MU, decompose=True) # note that the last femx is this one and thus for the wanted mu
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
            plt.plot(y_step, y, drawstyle='default', color='red', linewidth=2,label=labely)
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
        Mnorm = cp.array([]) # array to store the H1 norms of the FEM solutions
        for mu in p:
            x = ReducedOrder().FEMsolve(mu)
            Mnorm = cp.append(Mnorm,H1_norm_approx(x, h=1/N)) # compute the H1 norm of the FEM solution
        selected = [list(p[cp.argmax(Mnorm).get()])] # start with the point with the highest H1 norm
        return list(selected)
    elif VromPrev is not None:
        Mnorm = cp.array([])
        for mu in p:
            if list(mu) in VromPrev: #skip over mus in Vrom 
                Mnorm  = cp.append(Mnorm,-1) # adding zero wont affect it, because the errors will be >=0, if this was max the rest would also be zero. This keeps dimension of Mnorm similar to p
                continue
            rb = ReducedOrder(VromPrev, mu, boundary=0)
            xfem = rb.FEMsolve()
            xrb = rb.solve()
            Mnorm = cp.append(Mnorm, H1_norm_approx(xfem - xrb, h=1/N))  # compute the H1 norm of the difference
            index = cp.argmax(Mnorm).get()
        selected = [list(p[index])]  # select the point with the highest H1 norm
        return np.append(VromPrev, selected, axis=0), selected, Mnorm, index



def mixed_difference(x,h=1):
    """ use finite difference method to compute derivative of array for H1 norm approximation"""
    diffs = cp.empty_like(x)
    diffs[:-1] = x[1:] - x[:-1]  # forward differences
    diffs[-1] = x[-1] - x[-2]    # backward difference for last element
    return diffs/h

def H1_norm_approx(x,h=1):
    """
    Use trapezoidal rule to approximate the H1 norm of a 1D array.
    """
    x = cp.abs(x) # note absolute value
    diffs = mixed_difference(x, h)
    # Use trapezoidal rule for numerical integration
    integral1 = cp.trapz(diffs**2, dx=h)
    integral2 = cp.trapz(x**2, dx=h)
    return cp.asnumpy(cp.sqrt(integral1+integral2))


def error(mus, testP):
    listy=cp.array([]) # list to store the errors
    for mu in testP: # iterate over all mu's that are not in mus
        # print(" Computing error for mu:", mu, " at index", len(listy))
        mu = mu.tolist()
        rb = ReducedOrder(mus,mu, boundary=0)
        x= rb.solve()
        xfem = rb.FEMsolve()
        listy = cp.append(listy,H1_norm_approx(x-xfem, h=1/N))
    return cp.max(listy), listy

def ex3(alternative = False):
    blues = ['#87CEFA', '#1E90FF', '#4169E1', '#0000CD', '#000080']
    reds= ["#FF7640", '#F4A460', '#FF8C00', "#C97353", '#D2691E']
    for k in range(5):
        testP = P(alternative=alternative).random(size=500) # generate random points in the parameter space for testing
        error1=[]
        t = time.time()
        Vrom1 = P(alternative=alternative).random(size=50) # generate random point in the parameter space
        for n in range(2,51):
            print("Computing non-greedy ROM for n =", n, " time elapsed:", time.time() - t)
            t = time.time()
            error1.append(error(Vrom1[:n,:], testP)[0].get())
        
        
        error2=[] 
        pTrain = P(alternative=alternative).random(size=500) # generate random points in the parameter space for training
        Vrom1 = VromGreedy(pTrain,initial=True) # initialize the greedy ROM with the first point
        for n in range(2,51):
            print("Computing greedy ROM for n =", n, " time elapsed:", time.time() - t)
            t = time.time()
            Vrom1 = VromGreedy(pTrain, VromPrev=Vrom1)[0] # greedily add to Vrom
            error2.append(error(Vrom1, testP)[0].get())
        plt.plot(range(2,51), error1,color = blues[k] ,label="Non-Greedy ROM"+str(k))
        plt.plot(range(2,51), error2,color= reds[k], label="Greedy ROM"+str(k))
    plt.xlabel("Number of basis functions")
    plt.ylabel("Argmax H1 norm error")
    plt.title("Comparison of Greedy and Non-Greedy ROM")
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()


# ex3()
# ex3(True) #alternative True, alternative P


def test():
    PP = P(10).random(size=500)
    rb = ReducedOrder(PP, [0.02,0,1], boundary=0)
    x = rb.solve()
    x = rb.addBoundary(x)  # add boundary condition if needed
    xfem = rb.FEMsolve()
    xfem = rb.addBoundary(xfem)  # add boundary condition if needed
    rb.plotter(cp.asnumpy(x), cp.asnumpy(xfem), title="Reduced Order Model vs Finite Element Method", xlabel="x", ylabel="u(x)", labelx="ROM", labely="FEM")
    print("H1 norm error:", H1_norm_approx(x-xfem, h=1/N))
    print("error: ", error(PP, P(10).random(size=500))[0].get())
    plt.show()
# test()
def test2():
    # Test if ROM gives same result as FEM for the SAME parameter
    mu_test = [0.5, 0.3, 0.7]
    mus_basis = [[0.1, 0.2, 0.4], [0.8, 0.6, 0.9]]  # basis parameters
    # Create ROM with mu_test as the target parameter
    
    # See how close it is when using ROM as regular FEM
    # just use the test also as basis
    rom = ReducedOrder([mu_test], mu_test)
    # rom = ReducedOrder(mus_basis, mu_test)
    
    x_rom = rom.solve()  # ROM solution
    x_fem = rom.FEMsolve(mu_test)  # FEM solution for same parameter
    print("Error for same parameter:", H1_norm_approx(x_fem - x_rom, h=1/N))
    rom.plotter(cp.asnumpy(x_rom), cp.asnumpy(x_fem), title="ROM vs FEM for same parameter", xlabel="x", ylabel="u(x)", labelx="ROM", labely="FEM")
    plt.show()
test2()