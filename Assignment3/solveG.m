% This Matlab-function solves the nonlinear equation (12.136)
% required to solve the Riemann problem for the shallow water
% equations.
% solution method: Newton iteration

function solve = solveG(A,B);
maxit = 50;
tol = 1e-8;
k = 0;
z = 1;
G = g(z) + sqrt(A)*g(z/A) - B;
dG = dg(z) + dg(z/A)/sqrt(A);
conv = 0;
while (~conv)
  k = k+1;
  z = z - G/dG;
  G = g(z) + sqrt(A)*g(z/A) - B;
  dG = dg(z) + dg(z/A)/sqrt(A);
  conv = (abs(G)<tol) | (k==maxit);
end
solve = z;
