%% Part C

clear; clc; close all

% Parameters for synthetic signal to try
T = 2*pi; % Total time interval
omega = 2.0; % Frequency
phi = pi/6; % Phase

% Rewrite the SHO equation as a first-order matrix system Y' = AY
% State vector Y = [y; y']
A = [0 1;-omega^2 0];

% Initial conditions from analytical solution
Y0 = [sin(phi);omega*cos(phi)];

h = [0.2,0.1,0.05,0.025,0.0125];

% SELECT VALUE OF TIME STEP TO PLOT HERE
h2plot = 0.2;

max_euler_Err = zeros(length(h),1);
max_RK_Err = zeros(length(h),1);

for i = 1:length(h) 
    
    t = 0:h(i):T;
    N = length(t);

    euler_Y = euler_simulate(h(i),N,A,Y0);
    RK_Y = RK_simulate(h(i),N,A,Y0);   
    
    % Calculate maximum absolute global error over the integration interval
    analytic_y = sin(omega*t + phi);

    euler_Err = euler_Y(1,:) - analytic_y;
    RK_Err = RK_Y(1,:) - analytic_y;
    
    max_euler_Err(i) = max(abs(euler_Err));
    max_RK_Err(i) = max(abs(RK_Err));
    
    % Plot numerical and analytical solutions for selected step size
    if h(i) == h2plot
        t = 0:h(i):T;
        
        hold on
        plot(t,analytic_y)
        plot(t,euler_Y(1,:))
        plot(t,RK_Y(1,:))

        legend('Analytical solution','Euler','RK4')
        title(['SHO analytical solution vs Euler and RK4 with step size h = ',num2str(h(i))])
        xlabel('Time')
        ylabel('Displacement')
    end
end

% Compare convergence of maximum global error with step size
figure
hold on
plot(h,max_euler_Err)
plot(h,max_RK_Err)

% Estimate convergence order using linear regression on log-log data
p_euler = polyfit(log(h), log(max_euler_Err'), 1);
p_RK = polyfit(log(h), log(max_RK_Err'), 1);

order_euler = p_euler(1);
order_RK = p_RK(1);

grid on
set(gca, 'XScale', 'log', 'YScale', 'log')
legend(sprintf('Euler (order = %.2f)', order_euler), sprintf('RK4 (order = %.2f)', order_RK), 'Location', 'best')
title('Maximum Global Error against Step Size')
xlabel('Step Size h')
ylabel('Maximum Absolute Global Error')


% Fourth-order Runge-Kutta integration
function Y = RK_simulate(dt,N,A,Y0)

Y = zeros(2,N);
Y(:,1) = Y0;

for i = 1:N-1

    K1 = dt*A*Y(:,i);
    K2 = dt*A*(Y(:,i) + K1/2);
    K3 = dt*A*(Y(:,i) + K2/2);
    K4 = dt*A*(Y(:,i) + K3);

    Y(:,i+1) = Y(:,i) + 1/6 * (K1 + 2*K2 + 2*K3 + K4);
end
end

% Forward Euler integration
function Y = euler_simulate(dt,N,A,Y0)

Y = zeros(2,N);
Y(:,1) = Y0;

for i = 1:N-1
    Y(:,i+1) = (eye(2) + A*dt) * Y(:,i);
end
end

