import numpy as np
from typing import Callable, Any
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
class thetaMethod:
    def __init__(self, theta, A, b: Callable[[float], Any], init: np.ndarray, del_t =1/10, timeLength=100):
        """
        theta: variable for theta method
        A: A matrix for iterating u
        b: b vector, time depended function, is callable python function
        init: initial vector u_1
        del_t : delta T, as in time 
        timeLength: number of iteration steps over time 
        """
        self.theta=  theta
        self.A = A
        self.b = b
        self.init = init
        self.N_x = len(init)
        self.del_t = del_t
        self.TimeLength = timeLength
    def __iter__(self):
        n,un = (1, self.init)
        self.TimeMatrix = np.zeros((self.N_x,self.TimeLength))
        self.TimeMatrix[:,n-1] = un
        yield n,un
        while n<self.TimeLength:
            un = np.linalg.solve((np.identity(self.N_x) - self.theta*self.del_t*self.A), (np.identity(self.N_x)+(1-self.theta)*self.del_t*self.A)@un+
                                self.del_t*((1-self.theta)*self.b(self.del_t*n) + self.theta*self.b(self.del_t*(n+1))))
            n+=1
            self.TimeMatrix[:,n-1] = un
            yield (n,un)
    def solve(self):
        """ if not iterated over solution this will do"""
        n,un = (1, self.init)
        self.TimeMatrix = np.zeros((self.N_x,self.TimeLength))
        self.TimeMatrix[:,n-1] = un
        while n<self.TimeLength:
            un = np.linalg.solve((np.identity(self.N_x) - self.theta*self.del_t*self.A), (np.identity(self.N_x)+(1-self.theta)*self.del_t*self.A)@un+
                                self.del_t*((1-self.theta)*self.b(self.del_t*n) + self.theta*self.b(self.del_t*(n+1))))
            n+=1
            self.TimeMatrix[:,n-1] = un
        return self.TimeMatrix
    def plotHeat(self):
        xticks = np.round(np.linspace(0,self.TimeLength*self.del_t, self.TimeLength), decimals=3)
        yticks = np.round(np.linspace(0,1,self.N_x), decimals=3)
        ax = sns.heatmap(self.TimeMatrix, norm = LogNorm(), linewidth=0.5, xticklabels=xticks, yticklabels=yticks)
        plt.xlabel("Time")
        plt.ylabel("Radius")
        plt.show()
class HeatTransfer:
    def __init__(self, N_r, R=1, Bi=1, omega=1):
        self.N_r = N_r
        self.R = R
        self.Bi = Bi
        self.omega = omega
        self.__initialise()
    def __initialise(self):
        self.delta_r = self.R/(self.N_r-1)
        r = np.linspace(0, self.R, self.N_r)        
        r_j =r[1:-1]

        A_diag = np.zeros(self.N_r)
        A_upper  = np.zeros(self.N_r-1)
        A_lower  = np.zeros(self.N_r-1)
        
        r_ph_sq = np.square(r_j + self.delta_r/2)
        r_mh_sq = np.square(r_j - self.delta_r/2)
        A_diag[1:-1] =-(r_ph_sq + r_mh_sq) / np.square(r_j)
        A_upper[1:] = r_ph_sq / np.square(r_j)
        A_lower[:-1] = r_mh_sq / np.square(r_j)

        A = np.diag(A_diag) + np.diag(A_upper, k=1) + np.diag(A_lower, k=-1)

        A[0,0] = -2
        A[0,1] = 2
        A[self.N_r-1, self.N_r-2]= 1
        A[self.N_r-1, self.N_r-1] =  -(1 + self.Bi * self.delta_r)#-(2*self.Bi*self.delta_r**2 + 1+self.Bi*self.delta_r)

        A = A/self.delta_r**2
        def b(t):
            b = np.zeros(self.N_r)
            b[-1] = np.sin(self.omega*t)*(2*self.Bi + self.Bi/self.delta_r)
            return b
        self.matrixA = A
        self.vectorb = b

np.set_printoptions(linewidth=200)
N_x = 5
N_t = 100
classs = HeatTransfer(N_x, Bi=0.2, omega=0)
init =np.zeros(N_x)
init[0]=30
solver = thetaMethod(0.5, classs.matrixA, classs.vectorb, init,del_t=0.1, timeLength=N_t)
for (n,un) in solver:
    print("step ",n,":  ",un)
# solver.solve()
solver.plotHeat()
print(classs.matrixA)
