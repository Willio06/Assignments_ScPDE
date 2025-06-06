%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  This Matlab-script computes a numerical solution of the shallow water eq.
%  Numerical method: Godunov scheme
%
%  meaning of (most important) variables
%  Nx       number of grid points
%  Nt       number of time steps
%  dltx     grid size
%  dltt     time step
%  x        (spatial) grid
%  t        time
%  h        water height
%  phi      geopotential
%  u        flow velocity
%  u2       u2 = phi*u
%  flux1    first component of numerical flux
%  flux2    second component of numerical flux
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; format long e; 

%  (physical) parameters
L = 10; gg = 9.81;

%  numerical parameters and spatial grid
Nx = input('number of grid points: ');
dltx = (2*L)/(Nx-1);
for j=1:Nx
  x(j) = -L + (j-1)*dltx;
end
tmax = input('final time: ');
Nt = input('number of time steps: ');
dltt = tmax/Nt; tau = dltt/dltx;

%  initial condition
t = 0.0;
for j=1:Nx
  h(j) = 1;
  if ( (x(j)>=-1) & (x(j)<=1) )
    u(j) = 2;
  else
    u(j) = 1;
  end
  phi(j) = gg*h(j); c(j) = sqrt(phi(j));
  u2(j) = phi(j)*u(j);
end

% time integration
M = moviein(Nt);
for n=1:Nt
  t = t+dltt;

% computation flux
  for j=1:Nx-1

% solution Riemann problems
    phir = phi(j+1); phil = phi(j);
    ur = u(j+1); ul = u(j);
       .
       .
       .
     
% determine similarity solution

% two shocks
    if (B<sqrt(A)*g(1/A)) & (B<g(A))
       .
       .
       .
    end

% 1-rarefaction wave and 2-shock
    if (B>=sqrt(A)*g(1/A)) & (B<g(A))
       .
       .
       .
    end

% 1-shock and 2-rarefaction wave
    if (B<sqrt(A)*g(1/A)) & (B>=g(A))
       .
       .
       .
    end

% two rarefaction waves
    if (B>=sqrt(A)*g(1/A)) & (B>=g(A))
       .
       .
       .
    end

% expression for the flux
    flux1(j) = ... ;
    flux2(j) = ... ;

  end
    
% update
  for j=2:Nx-1
    phi(j) = phi(j) - tau*( flux1(j)-flux1(j-1) ); h(j) = phi(j)/gg;
    u2(j) = u2(j) - tau*( flux2(j)-flux2(j-1) ); u(j) = u2(j)/phi(j);
  end
 
% plot results
  subplot( 2,1,1 );
  plot( x,h,'b','LineWidth',1.6 );
  grid; axis( [-L L 0.5 1.5 ] );
  grid;
  xlabel( 'x','FontSize',16 );
  ylabel( 'h','FontSize',16,'Rotation',0 );
  subplot( 2,1,2 );
  plot( x,u,'b','LineWidth',1.6 );
  grid; axis( [-L L 0.5 2.5] );
  xlabel( 'x','FontSize',16 );
  ylabel( 'u','FontSize',16,'Rotation',0 );
  M(:,n) = getframe;
end
