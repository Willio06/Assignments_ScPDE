% wave2d_staggered_leapfrog.m
% Staggered Leapfrog Scheme for 2D Wave Equation in a Basin with Visualization

clear; clc; close all;

%% Parameters
a = 1;              % wave speed
alpha = 1;          % initial Gaussian decay
L = 2;              % domain extends from -L to L in x and y
Nx = 101;           % number of grid points in x
Ny = 101;           % number of grid points in y
dx = 2*L/(Nx-1);    % grid spacing
dy = dx;            % square grid
dt = 0.4 * dx / a;  % time step satisfying stability condition (CFL < 1/sqrt(2))
T = 2;              % final time
Nt = round(T/dt);   % number of time steps
CFL = a*dt/dx;

fprintf("Running with CFL = %.4f\n", CFL);

%% Grid
x = linspace(-L, L, Nx);
y = linspace(-L, L, Ny);
[X, Y] = meshgrid(x, y);

%% Initial condition: u(x,y,0) = exp(-alpha(x^2 + y^2))
u = exp(-alpha * (X.^2 + Y.^2));
u_old = u;  % since du/dt = 0 initially

%% Initialize p and q
p = zeros(Nx-1, Ny);      % p on horizontal edges (i+1/2,j)
q = zeros(Nx, Ny-1);      % q on vertical edges (i,j+1/2)

% Compute initial p and q using central differences
for i = 1:Nx-1
    for j = 1:Ny
        p(i,j) = 0.5 * a * dt/dx * (u(i+1,j) - u(i,j));
    end
end

for i = 1:Nx
    for j = 1:Ny-1
        q(i,j) = 0.5 * a * dt/dy * (u(i,j+1) - u(i,j));
    end
end

%% Visualization setup
fig = figure('Visible','on'); % Create figure once
frame_count = 1;

%% Time stepping
for n = 1:Nt
    % Update u using current p and q
    for i = 2:Nx-1
        for j = 2:Ny-1
            u(i,j) = u_old(i,j) - ...
                a*dt/dx * (p(i,j) - p(i-1,j)) - ...
                a*dt/dy * (q(i,j) - q(i,j-1));
        end
    end

    % Enforce boundary condition: u = 0 at boundary
    u(1,:) = 0; u(end,:) = 0;
    u(:,1) = 0; u(:,end) = 0;

    % Update p using new u
    for i = 1:Nx-1
        for j = 2:Ny-1
            p(i,j) = p(i,j) - a*dt/dx * (u(i+1,j) - u(i,j));
        end
    end

    % Update q using new u
    for i = 2:Nx-1
        for j = 1:Ny-1
            q(i,j) = q(i,j) - a*dt/dy * (u(i,j+1) - u(i,j));
        end
    end

    drawnow;  % Force MATLAB to update the figure
    figure(fig);   % Focus on existing figure
    clf;           % Clear previous plots

    % Surface plot
    subplot(1,2,1);
    surf(x, y, u', 'EdgeColor', 'none');
    axis([-L L -L L -0.5 1]);
    xlabel('x'); ylabel('y'); zlabel('u');
    title(sprintf('Surface Plot: t = %.2f', n*dt));
    view(45, 30);
    colormap(jet);

    % Heatmap
    subplot(1,2,2);
    imagesc(x, y, u');
    axis equal tight;
    colorbar;
    xlabel('x'); ylabel('y');
    title(sprintf('Heatmap: t = %.2f', n*dt));
    set(gca, 'YDir', 'normal');  % Correct y-axis orientation
    
    % Save image every 3 steps
    if mod(n, 3) == 0 || n == Nt
        if ~exist('wave_a1', 'dir')
            mkdir('wave_a1');
        end
        filename = sprintf('wave_a1/wave_%d.png', frame_count);
        saveas(fig, filename);
        frame_count = frame_count + 1;
    end

    % Update u_old
   u_old=u;
end