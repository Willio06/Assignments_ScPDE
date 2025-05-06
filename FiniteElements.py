import numpy as np
import matplotlib.pyplot as plt

class FiniteElements:
    # define matrices for the uniform grid

    def __init__(self, n, a,b,mu, boundary=0):
        self.N = n
        self.start = a
        self.end = b
        self.mu = mu
        h = (b-a)/n    
        self.h = h
        B1 = (1/h)*(np.diag((n-1)*[2]) + np.diag((n-2)*[-1], 1)+ np.diag((n-2)*[-1], -1))
        B2 = (h)*(np.diag((n-1)*[2/3]) + np.diag((n-2)*[1/6], 1)+ np.diag((n-2)*[1/6], -1))
        B3 = np.diag((n-2)*[1/2], 1)+ np.diag((n-2)*[-1/2], -1)

        A = mu[0]*B1 + mu[1]*B2 + mu[2]*B3
        b = h*np.ones(n-1)
        self.A = A
        self.b = b
        self.boundary = boundary
        
    def solve(self):
        # Placeholder for solving the finite element problem
        x  = np.linalg.solve(self.A, self.b)
        x = np.append(x, self.boundary)  # Append the boundary condition at the end
        x = np.insert(x, 0,self.boundary)
        self.x = x

        return x
    
    def visualize(self):
        # Build step-style x and y values for piecewise constant line
        x_step = np.arange(self.start, self.end+self.h, self.h)
        # Plot
        plt.plot(x_step, self.x, drawstyle='default', color='blue', linewidth=2)
        plt.xlim(self.start, self.end)
        plt.ylim(0, max(self.x) * 1.1)
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Piecewise constant line plot of vector x')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

prob = FiniteElements(10, 0, 1, [1, 0, 1])
sol = prob.solve()
prob.visualize()