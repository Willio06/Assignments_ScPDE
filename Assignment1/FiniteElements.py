import numpy as np
import matplotlib.pyplot as plt

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
        plt.ylim(0, max(self.x) * 1.1)
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Piecewise constant line plot of vector x')
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
        B3 = np.diag((n-2)*[1/2], 1)+ np.diag((n-2)*[-1/2], -1)
        A = mu[0]*B1 + mu[1]*B2 + mu[2]*B3
        b = h*np.ones(n-1)
        self.A = A
        self.b = b

n = 10
prob = problem1(n, 0, 1, [1, 0, 1])
prob = FiniteElements_1D(0,1,prob.A, prob.b, boundary=0, added_boundary=False)
sol = prob.solve()
prob.visualize()