% This Matlab-function computes the function g(z) from (12.121)
% occuring in the Riemann problem for the shallow water equations.
function gfunc = g(z);
if z > 1
  gfunc = 0.5*sqrt(2)*(1-z)*sqrt(1+1/z);
else 
  gfunc = 2*(1-sqrt(z));
end      
