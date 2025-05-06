import numpy as np
import matplotlib.pyplot as plt

# FEM solver

# required input
mu = [1, 0, 1];
N = 10;
h = 1/N

# output u

# define matrices for the uniform grid
B1 = (1/h)*(np.diag((N-1)*[2]) + np.diag((N-2)*[-1], 1)+ np.diag((N-2)*[-1], -1));
B2 = (h)*(np.diag((N-1)*[2/3]) + np.diag((N-2)*[1/6], 1)+ np.diag((N-2)*[1/6], -1));
B3 = np.diag((N-2)*[1/2], 1)+ np.diag((N-2)*[-1/2], -1);

A = mu[0]*B1 + mu[1]*B2 + mu[2]*B3
print(A)
b = h*np.ones(N-1)

x = np.linalg.solve(A, b)

plt.plot(x)
plt.show()
