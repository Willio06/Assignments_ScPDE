% This Matlab-function computes the derivative of the function
% g(z) in (12.121).
function dgfunc = dg(z);
if z > 1
  dgfunc = -0.25*sqrt(2)*( 2*(z^2)+z+1 )/(sqrt(1+1/z)*(z^2));
else 
  dgfunc = -1/sqrt(z);
end
