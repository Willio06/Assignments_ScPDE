import numpy as np
from typing import Callable, Any
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.linalg import expm
class thetaMethod:
    def __init__(self, theta, A, b: Callable[[float], Any], init: np.ndarray, del_t =1/10, TimeElements=100):
        """
        theta: variable for theta method
        A: A matrix for iterating u
        b: b vector, time depended function, is callable python function
        init: initial vector u_1
        del_t : delta T, as in time 
        TimeElements: number of iteration steps over time 
        """
        self.theta=  theta
        self.A = A
        self.b = b
        self.init = init
        self.N_x = len(init)
        self.del_t = del_t
        self.TimeElements = TimeElements
        self.TimeLength = int(TimeElements*del_t)
    def __iter__(self):
        n,un = (1, self.init)
        self.TimeMatrix = np.zeros((self.N_x,self.TimeElements))
        self.TimeMatrix[:,n-1] = un
        yield n,un
        while n<self.TimeElements:
            un = np.linalg.solve((np.identity(self.N_x) - self.theta*self.del_t*self.A), (np.identity(self.N_x)+(1-self.theta)*self.del_t*self.A)@un+
                                self.del_t*((1-self.theta)*self.b(self.del_t*n) + self.theta*self.b(self.del_t*(n+1))))
            n+=1
            self.TimeMatrix[:,n-1] = un
            yield (n,un)
    def solve(self):
        """ if not iterated over solution this will do"""
        n,un = (1, self.init)
        self.TimeMatrix = np.zeros((self.N_x,self.TimeElements))
        self.TimeMatrix[:,n-1] = un
        while n<self.TimeElements:
            un = np.linalg.solve((np.identity(self.N_x) - self.theta*self.del_t*self.A), (np.identity(self.N_x)+(1-self.theta)*self.del_t*self.A)@un+
                                self.del_t*((1-self.theta)*self.b(self.del_t*n) + self.theta*self.b(self.del_t*(n+1))))
            n+=1
            self.TimeMatrix[:,n-1] = un
        return self.TimeMatrix
    def plotHeat(self):
        xticks = np.round(np.linspace(0,self.TimeLength, self.TimeElements), decimals=3)
        xticks = np.where(np.arange(len(xticks)) % int(self.TimeElements/50) != 0, "", xticks.astype(str)) #print just 50 ticks on x-axis
        yticks = np.round(np.linspace(0,1,self.N_x), decimals=3)
        if self.N_x > 10:
            yticks = np.where(np.arange(len(yticks)) % int(self.N_x/10) != 0, "", yticks.astype(str))        
        ax = sns.heatmap(self.TimeMatrix, linewidth=0, xticklabels=xticks, yticklabels=yticks, cmap = "magma")
        plt.xlabel("Time")
        plt.ylabel("Radius")
        plt.title("Heat Transfer in a Sphere, Theta Method for theta= "+str(self.theta))
        # plt.show()
class ExpRungeKutta:
    def __init__(self, A, b: Callable[[float], Any], init: np.ndarray, del_t =1/10, TimeElements=100):
        """
        A: A matrix for iterating u
        b: b vector, time depended function, is callable python function
        init: initial vector u_1
        del_t : delta T, as in time 
        TimeElements: number of iteration steps over time 
        """
        self.A = A
        self.b = b
        self.init = init
        self.N_x = len(init)
        self.del_t = del_t
        self.TimeElements = TimeElements
        self.TimeLength = int(TimeElements*del_t)

        self.expA = expm(self.A*del_t)
        self.phi1 = np.linalg.inv(del_t* self.A)@(self.expA - np.identity(self.N_x))
        self.phi2 = np.linalg.inv(del_t* self.A)@(self.phi1 - np.identity(self.N_x))
    def __iter__(self):
        n,un = (1, self.init)
        self.TimeMatrix = np.zeros((self.N_x,self.TimeElements))
        self.TimeMatrix[:,n-1] = un
        yield n,un
        while n<self.TimeElements:
            un = self.expA@un + self.del_t*self.phi1@self.b(self.del_t*n) + self.del_t*self.phi2@(self.b(self.del_t*(n+1))- self.b(self.del_t*n))
            n+=1
            self.TimeMatrix[:,n-1] = un
            yield (n,un)
    def solve(self):
        """ if not iterated over solution this will do"""
        n,un = (1, self.init)
        self.TimeMatrix = np.zeros((self.N_x,self.TimeElements))
        self.TimeMatrix[:,n-1] = un
        while n<self.TimeElements:
            un = self.expA@un + self.del_t*self.phi1@self.b(self.del_t*n) + self.del_t*self.phi2@(self.b(self.del_t*(n+1))- self.b(self.del_t*n))
            n+=1
            self.TimeMatrix[:,n-1] = un
        return self.TimeMatrix
    def plotHeat(self):
        xticks = np.round(np.linspace(0,self.TimeLength, self.TimeElements), decimals=3)
        xticks = np.where(np.arange(len(xticks)) % int(self.TimeElements/50) != 0, "", xticks.astype(str)) #print just 50 ticks on x-axis
        yticks = np.round(np.linspace(0,1,self.N_x), decimals=3)
        if self.N_x > 10:
            yticks = np.where(np.arange(len(yticks)) % int(self.N_x/10) != 0, "", yticks.astype(str))
        ax = sns.heatmap(self.TimeMatrix, linewidth=0, xticklabels=xticks, yticklabels=yticks, cmap = "magma")
        plt.xlabel("Time")
        plt.title("Heat Transfer in a Sphere, Exponential Runge-Kutta Method")
        plt.ylabel("Radius")
        # plt.show()
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
        A[self.N_r-1, self.N_r-1] =  -(1 + self.Bi * self.delta_r + 2*self.Bi*self.delta_r**2)

        A = A/self.delta_r**2
        def b(t):
            b = np.zeros(self.N_r)
            b[-1] = np.sin(self.omega*t)*(2*self.Bi + self.Bi/self.delta_r)
            return b
        self.matrixA = A
        self.vectorb = b

np.set_printoptions(linewidth=200)
def test(Bi=10000,omega=2,Nx=3,Nt=100, delt=0.1):
    classs = HeatTransfer(Nx, Bi=Bi, omega=omega)
    print(classs.matrixA)

    solver = thetaMethod(0.5, classs.matrixA, classs.vectorb, np.zeros(Nx),del_t=delt, TimeElements=Nt)
    for (n,un) in solver:
        print("step ",n,":  ",un)
    solver.solve()
    solver.plotHeat()
    plt.show()
# test()

def generalTest(Bi=10000,omega=2,Nx=3,Nt=1000, delt=0.1):
    classs = HeatTransfer(Nx, Bi=Bi, omega=omega)
    # print(classs.matrixA)

    plt.subplot(1,2,1)
    solver = thetaMethod(0.5, classs.matrixA, classs.vectorb, np.zeros(Nx),del_t=delt, TimeElements=Nt)
    # for (n,un) in solver:
    #     print("step ",n,":  ",un)
    solver.solve()
    solver.plotHeat()

    plt.subplot(1,2,2)
    solverExp = ExpRungeKutta(classs.matrixA, classs.vectorb, np.zeros(Nx), del_t=delt, TimeElements=Nt)
    solverExp.solve()
    solverExp.plotHeat()
    plt.show() 

def MethodOrder(N_x=3):

    classs = HeatTransfer(N_x, Bi=1, omega=1)
    P = []
    print("-------------- Exponential Runge-Kutta Method Order --------------")
    for n in np.arange(10,1000,10):
        xn = ExpRungeKutta(classs.matrixA, classs.vectorb, np.zeros(N_x), del_t=10/n, TimeElements=n).solve()
        x2n = ExpRungeKutta(classs.matrixA, classs.vectorb, np.zeros(N_x), del_t=10/(2*n), TimeElements=2*n).solve()
        x4n = ExpRungeKutta(classs.matrixA, classs.vectorb, np.zeros(N_x), del_t=10/(4*n), TimeElements=4*n).solve()
        a = x2n[:,-1] - xn[:,-1]
        b = x4n[:,-1] - x2n[:,-1]
        p=np.log2(np.divide(a, b))
        P.append(np.mean(p))
        # print("n=", n, " Order=", np.mean(p))
    plt.plot(np.arange(10,1000,10), P, label="Exponential Runge-Kutta Method Order")
    for theta in [0,0.5,1]:
        print("-------------- Theta Method Order for theta=", theta, "--------------")
        P = []
        for n in np.arange(10,1000,10):
            xn = thetaMethod(theta, classs.matrixA, classs.vectorb, np.zeros(N_x), del_t=10/n, TimeElements=n).solve()
            x2n = thetaMethod(theta, classs.matrixA, classs.vectorb, np.zeros(N_x), del_t=10/(2*n), TimeElements=2*n).solve()
            x4n = thetaMethod(theta, classs.matrixA, classs.vectorb, np.zeros(N_x), del_t=10/(4*n), TimeElements=4*n).solve()
            a = x2n[:,-1] - xn[:,-1]
            b = x4n[:,-1] - x2n[:,-1]
            p=np.log2(np.divide(a, b))
            P.append(np.mean(p))
            # print("n=", n, " Order=", np.mean(p))
        plt.plot(np.arange(10,1000,10), P, label="Theta Method Order for theta="+str(theta))
    plt.xlabel("Number of Time elements n")
    plt.ylabel("Order of Convergence")
    plt.ylim(0.5, 2.5)
    plt.legend()
    plt.title("Order of Convergence for different Methods")
    plt.show()

generalTest()
MethodOrder()
