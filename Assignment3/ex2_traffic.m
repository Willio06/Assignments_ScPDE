clear; clc;

% Parameters
Nx = 200;
x = linspace(0, 1, Nx)';
dx = x(2) - x(1);
t_end = 1.0;
alpha = 1.5;
CFL = 0.9;
case_name = "shock"; % Options: "shock" or "rarefaction"

% Save plot settings
save_frames = true;         
save_every = 5; % Save every Nth frame
save_dir = 'traffic_shock_a1.5'; % Folder name
frame_count = 0;                 % Counter for saved frames

% Flux and derivatives
f  = @(q) q .* exp(alpha * q);
df = @(q) (1 + alpha * q) .* exp(alpha * q);

% Initial conditions
switch case_name
    case "shock"
        x0 = 0.2; ql = -1.0; qr = 0.0;
    case "rarefaction"
        x0 = 0.7; ql = -0.5; qr = -1.0;
end
q0 = ql * ones(size(x));
q0(x > x0) = qr;

% Time stepping
dt = CFL * dx / max(abs(df(q0)));
Nt = ceil(t_end / dt);
dt = t_end / Nt;
t_vec = linspace(0, t_end, Nt+1);

% Storage arrays
q_god = zeros(Nx, Nt+1);
q_muscl = zeros(Nx, Nt+1);
q_god(:,1) = q0;
q_muscl(:,1) = q0;

% === Minmod limiter function ===
minmod = @(a, b) 0.5 * (sign(a) + sign(b)) .* min(abs(a), abs(b)) .* (sign(a .* b) > 0);

% === First-order Godunov solver ===
q = q0;
for n = 1:Nt
    qL = q(1:end-1);
    qR = q(2:end);
    flux = zeros(size(qL));

    for i = 1:length(qL)
        ql = qL(i); qr = qR(i);
        if ql > qr
            s = (f(ql) - f(qr)) / (ql - qr);
            flux(i) = f(ql) * (s >= 0) + f(qr) * (s < 0);
        else
            if df(ql) >= 0
                flux(i) = f(ql);
            elseif df(qr) <= 0
                flux(i) = f(qr);
            else
                flux(i) = f(-1/alpha);
            end
        end
    end

    q(2:end-1) = q(2:end-1) - dt/dx * (flux(2:end) - flux(1:end-1));
    q(1) = q(2); q(end) = q(end-1);
    q_god(:,n+1) = q;
end

% === MUSCL-Godunov solver ===
q = q0;
for n = 1:Nt
    dq = zeros(size(q));
    dq(2:end-1) = minmod(q(2:end-1) - q(1:end-2), q(3:end) - q(2:end-1));

    qL = q(1:end-1) + 0.5 * dq(1:end-1);
    qR = q(2:end) - 0.5 * dq(2:end);
    flux = zeros(size(qL));

    for i = 1:length(qL)
        ql = qL(i); qr = qR(i);
        if ql > qr
            s = (f(ql) - f(qr)) / (ql - qr);
            flux(i) = f(ql) * (s >= 0) + f(qr) * (s < 0);
        else
            if df(ql) >= 0
                flux(i) = f(ql);
            elseif df(qr) <= 0
                flux(i) = f(qr);
            else
                flux(i) = f(-1/alpha);
            end
        end
    end

    q(2:end-1) = q(2:end-1) - dt/dx * (flux(2:end) - flux(1:end-1));
    q(1) = q(2); q(end) = q(end-1);
    q_muscl(:,n+1) = q;
end

% === Animation Plot ===
figure;
for n = 1:Nt+1
    plot(x, q_god(:,n), 'b', 'LineWidth', 1.6); hold on;
    plot(x, q_muscl(:,n), 'r', 'LineWidth', 1.6);
    legend('Godunov (1st order)', 'MUSCL (minmod)', 'Location', 'southeast');
    axis([0 1 -1.1 0.1]);
    xlabel('x', 'FontSize', 14);
    ylabel('q', 'FontSize', 14);
    title(sprintf('q(x,t) at t = %.3f', t_vec(n)), 'FontSize', 12);
    grid on;
    hold off;
    
    % Create directory if needed
    if save_frames && n == 1 && ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    
    % Save frame if it's the first or every Nth
    if save_frames && (n == 1 || mod(n-1, save_every) == 0)
        frame_count = frame_count + 1;
        frame_idx = (n - 1);  % 0-based index for t = 0
        filename = sprintf('%s/frame_%03d.png', save_dir, frame_count);
        saveas(gcf, filename);
    end

    drawnow;
    M(n) = getframe(gcf); %#ok<SAGROW>
end
