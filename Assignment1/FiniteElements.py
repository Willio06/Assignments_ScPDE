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
    
    def visualize(self):
        # Build step-style x and y values for piecewise constant line
        x_step = np.arange(self.start, self.end+self.h/2, self.h)# + self.h/2 such that the end is included, without extra point
        # Plot
        plt.plot(x_step, self.x, drawstyle='default', color='blue', linewidth=2)
        plt.xlim(self.start, self.end)
        # plt.ylim(0, max(self.x) * 1.1)
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Piecewise constant line plot of vector x')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    def comparePlot(self, real):
        try:
            self.x
        except AttributeError as e:
            print("\033[91mERROR: No solution found, please run solve() first.\033[0m")
        x_step = np.arange(self.start, self.end+self.h/2, self.h)# + self.h/2 such that the end is included, without extra point
        # Plot
        plt.plot(x_step, self.x, drawstyle='default', color='blue', linewidth=2, label='FEM Solution')
        plt.xlim(self.start, self.end)
        # plt.ylim(0, max(self.x) * 1.1)
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('FEM Solution vs Real Solution')
        plt.plot(x_step, real, drawstyle='default', color='red', linewidth=2, label='Real Solution')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

class problem1:
    def __init__(self, n, a, b, mu):
        self.start = a
        self.end = b
        self.mu = mu
        h = (b-a)/n
        B1 = (1/h)*(np.diag((n-1)*[2]) + np.diag((n-2)*[-1], 1)+ np.diag((n-2)*[-1], -1))
        B2 = (h)*(np.diag((n-1)*[2/3]) + np.diag((n-2)*[1/6], 1)+ np.diag((n-2)*[1/6], -1))
        B3 = np.diag((n-2)*[-1/2], 1)+ np.diag((n-2)*[1/2], -1)
        A = mu[0]*B1 + mu[1]*B2 + mu[2]*B3
        print(A)
        b = h*np.ones(n-1)
        self.A = A
        print(b)
        self.b = b
    def realSolution(self, x):
        if self.mu[2] == 0 or self.mu[0] == 0:
            raise Exception("Beta or Nu is 0, cannot compute real solution for this case")
        if np.any(0>x) or np.any(x>1):
            warnings.warn("\033[93m Warning: x is out of bounds\033[0m")
        return 1/(self.mu[2])*x - (np.exp(-self.mu[2]/self.mu[0] * x))/(self.mu[2]*(math.exp(-self.mu[2]/self.mu[0]) - 1)) + 1/(self.mu[2]*(math.exp(-self.mu[2]/self.mu[0]) - 1))

n = 5
prob = problem1(n, 0, 1, [1, 0, 1])
prob1 = FiniteElements_1D(0,1,prob.A, prob.b, boundary=0, added_boundary=False)

prob1.solve()
# prob1.visualize()
x = np.arange(0, 1+1/n, 1/n)
y = prob.realSolution(x)
prob1.comparePlot(y)
