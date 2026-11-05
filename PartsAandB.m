clear; clc; close all

% Parameters for synthetic signal
T = 2*pi; % Total time interval
dt = 0.001; % Base sampling step
omega = 2.0; % Frequency
phi = pi/6; % Phase
sigma = 0.01; % Noise standard deviation

t = 0:dt:T;

y_clean = sin(omega*t + phi);

dy_clean = omega*cos(omega*t + phi);

rng(1)
y_noisy = y_clean + sigma * randn(size(y_clean));

samples = length(t);

%% Part A

% Anonymous function to calculate the forward and backward derivative (each give same array for y)
backward_derivative = @(y,dt,k) (y(1+k:k:end) - y(1:k:end-k))/(k*dt);

% Anonymous function to calculate central derivative
central_derivative = @(y,dt,k) (y(1+2*k:k:end) - y(1:k:end-2*k))/(2*k*dt);

k_array = [1,2,4,8,16,64,128];

% RMSE calculated for each approximation scheme (RMSE instead of L2 used because number of
% points changes with k)
RMSE_backward = zeros(1,7);
RMSE_forward = zeros(1,7);
RMSE_central = zeros(1,7);

i = 1;

k2plot = 64;

for k = k_array
   
    dy_noisy_bwd = backward_derivative(y_noisy,dt,k);
    
    N = length(dy_noisy_bwd) - 1;
    
    % Backward (correction applied so only interior points used)
    RMSE_backward(i) = sqrt(1/N * sum((dy_noisy_bwd(1:end-1) - dy_clean(1+k:k:end-k)).^2));

    % Forward (correction applied so only interior points used)
    RMSE_forward(i) = sqrt(1/N * sum((dy_noisy_bwd(2:end) - dy_clean(1+k:k:end-k)).^2));

    % Central
    dy_noisy_cnt = central_derivative(y_noisy,dt,k);

    RMSE_central(i) = sqrt(1/N * sum((dy_noisy_cnt - dy_clean(1+k:k:end-k)).^2));

    i = i + 1;

    if k == k2plot

        backward_t = t(1+k:k:end);
        forward_t = t(1:k:end-k);
        central_t = t(1+k:k:end-k);

        hold on        
        plot(t,y_noisy)
        plot(t,dy_clean)        
        plot(backward_t,dy_noisy_bwd)
        plot(forward_t,dy_noisy_bwd)
        plot(central_t,dy_noisy_cnt)

        title(['Noisy y with numerical derivatives, subsample period k = ',num2str(k)])
        legend('noisy y','analytical derivative','backward derivative','forward derivative','central derivative')
        xlabel('Time')
        ylabel('Displacement')
    end
end

h = dt*k_array;

figure
hold on

plot(h, RMSE_backward)
plot(h, RMSE_forward)
plot(h, RMSE_central)

set(gca, 'XScale', 'log', 'YScale', 'log')
legend('Backward','Forward','Central')
title('RMSE error for numerical derivatives')
xlabel('Sampling period h')
ylabel('RMSE Error')

%% Part B
tau = (t - T/2)/(T/2); % Centred and scaled time

nMax = 10; % Test polynomial fits from order 1 to nMax
n2plot = 10;

condition = zeros(nMax,1);
RMSE_dy = zeros(nMax,1);
RMSE_validation = zeros(nMax,1);

train_space = 128; % Spacing between points used for training

train_idx = train_space:train_space:samples;
val_idx = setdiff(1:samples, train_idx);

t_train = tau(train_idx);
y_train = y_noisy(train_idx);

t_val = tau(val_idx);
y_val = y_clean(val_idx);

figure

for n = 1:nMax
    [p,condition(n)] = regression(n, t_train, y_train');   
    
    y_fitted = polyval(p, tau);
    
    dp = polyder(p);
    
    dy_fitted = polyval(dp, tau) * 2/T; % Derivative must be scaled back for comparison to original
    
    RMSE_dy(n) = sqrt(1/(samples-2) * sum((dy_fitted(2:end-1) - dy_clean(2:end-1)).^2)); % For comparison with central difference
    
    if train_space ~= 1
        RMSE_validation(n) = sqrt((train_space)/((train_space - 1)*samples) * sum((y_fitted(val_idx) - y_noisy(val_idx)).^2));
    end
    
    % Plot the chosen fitted polynomial and compare with the clean signals
    if n == n2plot
        hold on
        
        % If hold-out validation used, show training data
        if train_space ~= 1
            plot(t(train_idx),y_train,'x','MarkerSize', 7,'LineWidth', 1)
            legend('Training data', 'y', 'Fitted', 'dy/dx', 'dy/dx fitted')            
            title(['Order ',num2str(n),' polynomial fit, training points every ',num2str(train_space)])
        else
            legend('y','Fit to y','dy/dx','dy/dx from fit')
            title(['Order ',num2str(n),' polynomial fit to sine curve'])
        end

        plot(t, y_clean)
        plot(t, y_fitted)
        plot(t, dy_clean)
        plot(t, dy_fitted)
        
        xlabel('time')
    end
end

% Plot Condition Number against Polynomial Order
figure
plot(1:nMax,condition)

set(gca, 'YScale', 'log')
title('Condition Number against Polynomial Order')
xlabel('Polynomial Order')
ylabel('Condition Number')

% Plot RMSE to noiseless derivative of polynomial fits and central difference
figure
hold on

plot(1:nMax, RMSE_dy)
plot(1:nMax, RMSE_central(1)*ones(10,1))
plot(1:nMax, RMSE_central(7)*ones(10,1))

set(gca, 'YScale', 'log')
title('RMSE to noiseless derivative of polynomial fits and central difference')
legend("RMSE from polynomial approximation", "RMSE from central difference, k = 1", "RMSE from central difference, k = 128")
xlabel('Polynomial Order')
ylabel('RMSE from noiseless derivative')



% If validation used, plot RMSE to the original function against polynomial order
if train_space ~= 1
    figure
    hold on

    plot(1:nMax, RMSE_validation)

    set(gca, 'YScale', 'log')
    title(['Validation data RMSE Error (train on every ',num2str(train_space),' points)'])
    
    xlabel('Polynomial Order')
    ylabel('RMSE from noiseless signal')
end

function [x,condition] = regression(n,t,y)

V = t(:).^(n:-1:0); % Construct Vandermonde matrix

x = V\y; % Calculate polynomical coefficients with LS fitting

condition = cond(V'*V); % condition number using 2 norm

end
