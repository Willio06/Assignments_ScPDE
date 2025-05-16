import numpy as np
import matplotlib.pyplot as plt
import warnings
import math

class FiniteElements_1D:
    # given N, there are N+1 pounts in the mesh
    def __init__(self,a,b,A,B, boundary=0, added_boundary=True):
        self.N = np.shape(B)[0]+(-1)**int(added_boundary) 
        # if added_boundary is True, we add the boundary value to the system, and thus is in matrix A
        # with N from 0 to N, we have N+1 points in the mesh, ie A is (N+1)x(N+1),  ie N = shape(B)[0]-1
        
        # if added_boundary is False, we do not add the boundary value to the system
        # with N from 0 to N, we have N-1 inner points in the mesh, ie A is (N-1)x(N-1),  ie N = shape(B)[0]+1
        self.start = a
        self.end = b
        h = (b-a)/self.N    
        self.h = h
        self.A = A
        self.b = B
        self.boundary = boundary
        self.added_boundary = added_boundary
        
    def solve(self):
        # Placeholder for solving the finite element problem
        x  = np.linalg.solve(self.A, self.b)
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
    
    def visualize(self):
        try:
            self.x
        except AttributeError as e:
            print("\033[91mERROR: No solution found, please run solve() first.\033[0m")
        # Build step-style x and y values for piecewise constant line
        self.plotter(self.x, title=f'FEM Solution with n={self.N}', xlabel='X axis', ylabel='Y axis')
    def comparePlot(self, real):
        try:
            self.x
        except AttributeError as e:
            print("\033[91mERROR: No solution found, please run solve() first.\033[0m")
        self.plotter(self.x, real, title=f'FEM Solution with n={self.N} vs Real Solution', xlabel='X axis', ylabel='Y axis', labelx="FEM solution", labely="real solution")


class problem1:
    def __init__(self, n, a, b, mu):
        self.start = a
        self.end = b
        self.mu = mu
        h = (b-a)/n
        B1 = (1/h)*(np.diag((n-1)*[2]) + np.diag((n-2)*[-1], 1)+ np.diag((n-2)*[-1], -1))
        B2 = (h)*(np.diag((n-1)*[2/3]) + np.diag((n-2)*[1/6], 1)+ np.diag((n-2)*[1/6], -1))
        B3 = np.diag((n-2)*[1/2], 1)+ np.diag((n-2)*[-1/2], -1)
        A = mu[0]*B1 + mu[1]*B2 + mu[2]*B3
        # print(A)
        b = h*np.ones(n-1)
        self.A = A
        # print(b)
        self.b = b
    def realSolution(self, x):
        if self.mu[2] == 0 or self.mu[0] == 0:
            raise Exception("Beta or Nu is 0, cannot compute real solution for this case")
        if np.any(0>x) or np.any(x>1):
            warnings.warn("\033[93m Warning: x is out of bounds\033[0m")
        self.mu[0] = np.longdouble(self.mu[0])
        return 1/(self.mu[2])*x - (np.exp(self.mu[2]/self.mu[0] * x))/(self.mu[2]*(np.exp(self.mu[2]/self.mu[0]) - 1)) + 1/(self.mu[2]*(np.exp(self.mu[2]/self.mu[0]) - 1))
    def realDerivative(self,x):
        if self.mu[2] == 0 or self.mu[0] == 0:
            raise Exception("Beta or Nu is 0, cannot compute real solution for this case")
        if np.any(0>x) or np.any(x>1):
            warnings.warn("\033[93m Warning: x is out of bounds\033[0m")
        self.mu[0] = np.longdouble(self.mu[0])
        return 1/(self.mu[2])- (np.exp(self.mu[2]/self.mu[0] * x))/(self.mu[0]*(np.exp(self.mu[2]/self.mu[0]) - 1))



def exA():
    for nu in [10**(-3), 10**(-4)]:
        for i in range(4):
            n = [10,100,1000,10000][i]
            prob = problem1(n, 0, 1, [nu, 0, 1])
            prob1 = FiniteElements_1D(0,1,prob.A, prob.b, boundary=0, added_boundary=False)

            prob1.solve()
            # prob1.visualize()
            x = np.linspace(0, 1,100)
            y = prob.realSolution(x)
            plt.subplot(2,2,i+1)
            prob1.comparePlot(y)
        plt.show()

def exB():
    Errors1 =[]
    Errors2=[]
    m =20
    for n in np.linspace(10,10**4,m):
        n = int(n)
        print(n)
        prob = problem1(n, 0, 1, [1, 0, 1])
        prob1 = FiniteElements_1D(0,1,prob.A, prob.b, boundary=0, added_boundary=False)
        prob1.solve()
        Fem = prob1.x
        real = prob.realSolution(np.linspace(0, 1, n+1))
        realder = prob.realDerivative(np.linspace(0, 1, n+1))
        FEMder = np.append((Fem[1:] - Fem[:-1]) / prob1.h,(Fem[-1] - Fem[-2]) / prob1.h)   
        errorDer = np.square(np.abs(FEMder - realder))
        error = np.square(np.abs(Fem - real))
        NormL2 = np.sqrt((prob1.h / 2) * np.sum(error[:-1] + error[1:])) # inside trapezium rule for integrating
        NormH1 = np.sqrt((prob1.h/2 )* np.sum(errorDer[:-1] + errorDer[1:])) + NormL2**2
        Errors1.append(NormL2)
        Errors2.append(NormH1)
    plt.plot(np.linspace(10,10**4,m), Errors1,linewidth=2, label="L2 norm" )
    plt.plot(np.linspace(10,10**4,m), Errors2,linewidth=2 , label="H1 norm")

    plt.title("L2 en H1 norm of FEM approximation error")
    plt.xlabel("Size of N")
    plt.yscale('log')
    plt.legend()
    plt.show()

exB()