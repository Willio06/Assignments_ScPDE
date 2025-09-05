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
    A = phir/phil;
    B = (ur - ul)/sqrt(phil);
    z1 = solveG(A,B);
    z2 = z1/A;
    phistar = z1*phil;
    ustar = ul + g(z1)*sqrt(phil);
    s1 = ul - sqrt(phil*(1+z1)*z1/2);
    s2 = ur + sqrt(phir*(1+z2)*z2/2);
    cstar = sqrt(phistar);
    cl = sqrt(phil);
    cr = sqrt(phir);
     
% determine similarity solution

% two shocks
    if (B<sqrt(A)*g(1/A)) & (B<g(A))
       if s1>0
           u_flux = ul;
           phi_flux = phil;
       elseif s1 < 0 < s2
           u_flux = ustar;
           phi_flux = phistar;
       elseif s2 < 0
           u_flux = ur;
           phi_flux = phir;
       end
    end

% 1-rarefaction wave and 2-shock
    if (B>=sqrt(A)*g(1/A)) & (B<g(A))
       if ul-cl>0
           u_flux = ul;
           phi_flux = phil;
       elseif (ul-cl<0) & (0 < ustar - cstar)
           u_flux = ((ul + 2*cl)^2)/9;
           phi_flux = (ul + 2*cl)/3;
       elseif (ustar - cstar < 0) & (0< s2)
           u_flux = ustar;
           phi_flux = phistar;
       elseif s2>0
           u_flux = ur;
           phi_flux = phistar;
       end
    end

% 1-shock and 2-rarefaction wave
    if (B<sqrt(A)*g(1/A)) & (B>=g(A))
       if s1>0
           u_flux = ul;
           phi_flux = phil;
       elseif (s1<0) & (0<ustar + cstar)
           u_flux = ustar;
           phi_flux = phistar;
       elseif (ustar + cstar <0) & (0< ur + cr)
           u_flux = ((2*cr - ur)^2)/9;
           phi_flux = (ur - 2*cr)/3;
       elseif ur + cr <0
           u_flux = ur;
           phi_flux = phir;
       end
    end

% two rarefaction waves
    if (B>=sqrt(A)*g(1/A)) & (B>=g(A))
       if ul-cl>0
           u_flux = ul;
           phi_flux = phil;
       elseif (ul-cl <0)&(0< ustar - cstar)
           u_flux = ((ul + 2*cl)^2)/9;
           phi_flux = (ul + 2*cl)/3;
       elseif (ustar - cstar <0) & (0< ustar + cstar)
           u_flux = ustar;
           phi_flux = phistar;
       elseif (ustar + cstar <0) & (0<ur + cr )
           u_flux = ((2*cr - ur)^2)/9;
           phi_flux = (ur - 2*cr)/3;
       elseif ur+cr <0
           u_flux = ur;
           phi_flux =phir;
       end
    end

% expression for the flux
    flux1(j) = phi_flux*u_flux ;
    flux2(j) = phi_flux*u_flux^2 + phi_flux^2/2;

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
  title(sprintf('Water Height at t = %.3f', t), 'FontSize', 10);
  subplot( 2,1,2 );
  plot( x,u,'b','LineWidth',1.6 );
  grid; axis( [-L L 0.5 2.5] );
  xlabel( 'x','FontSize',16 );
  ylabel( 'u','FontSize',16,'Rotation',0 );
  title(sprintf('Water Velocity at t = %.3f', t), 'FontSize', 10);
  M(:,n) = getframe;

  % save every 25 frames
if mod(n, 25) == 0
    if ~exist('DamFrames', 'dir')
        mkdir('DamFrames');
    end
    frame_idx = n / 25;
    filename = sprintf('DamFrames/Dam_%d.png', frame_idx);
    saveas(gcf, filename);
end
end
