clear all; close all;

% Load the data
load('ps8_data.mat');

%% Define parameters
[N, D] = size(Xsim);
M = 1;

% Normalize Xsim (DxN matrix)
mu = mean(Xsim, 1);
X = (Xsim - mu);

% Initialize FA parameters
W = rand(D, M);
Psi = eye(D);
log_likelihood = [];
max_iter = 100;
% Run EM algorithm and update W and Psi
for iter = 1:max_iter
    % E-step
    inv_Psi = inv(Psi);
    G = inv(eye(M) + W' * inv_Psi * W);
    Ez = G * W' * inv_Psi * X';
    Ezz = G + Ez * Ez';
    
    % M-step
    W_new = X' * Ez' * inv(Ezz);
    Psi_new = 1/N * diag(diag(X' * X - W_new * Ez * X));
    % Update parameters
    W = W_new;
    Psi = Psi_new;
    
    % Calculate log-likelihood
    log_likelihood(iter) = -N*D/2*log(2*pi) - N/2*log(det(Psi + W'*W)) - 0.5 * trace(X * inv(Psi + W'*W) * X');
    
    
    % Check convergence
    if iter > 1 && abs(log_likelihood(iter) - log_likelihood(iter-1)) < 1e-4
        break;
    end
end

% Plot the log data likelihood versus EM iteration
figure;
plot(log_likelihood);
xlabel('EM iteration');
ylabel('Log data likelihood');
title('Log data likelihood vs EM iteration for FA');

% FA covariance
FA_cov = W * W' + diag(Psi);

% Check the covariances
sample_cov = cov(Xsim);
disp('Sample covariance:');
disp(sample_cov);
disp('FA covariance:');
disp(FA_cov);

% Plot the data points, mean, low-dimensional space, and projections
figure;
scatter(Xsim(:, 1), Xsim(:, 2), 'k.');
hold on;
scatter(mu(1), mu(2), 'go', 'LineWidth', 2);
proj_line = [mu' - 3 * W, mu' + 3 * W];
plot(proj_line(1, :), proj_line(2, :), 'k-');
for i = 1:N
x_n = Xsim(i, :);
z_n = W' * (x_n - mu)';
x_proj = (W * z_n)' + mu;
scatter(x_proj(1), x_proj(2), 'r.');
plot([x_n(1), x_proj(1)], [x_n(2), x_proj(2)], 'r-');
end
xlabel('x1');
ylabel('x2');
title('FA data points, mean, low-dimensional space, and projections');
legend({'Data points', 'Mean', 'Low-dimensional space', 'Projected points'}, 'Location', 'best');
hold off;