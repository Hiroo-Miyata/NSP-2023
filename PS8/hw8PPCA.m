clear all; close all;

% Load the data
load('ps8_data.mat');

%% Define parameters
[N, D] = size(Xsim);
M = 1;

% Xsim is assumed to be in 'ps8_data.mat'
% Xsim: 8x2 matrix of simulated data
mu = mean(Xsim, 1);
X = (Xsim - mu);


% Initialize PPCA parameters
W = rand(D, M);
sigma2 = 1;
log_likelihood = [];

% Define the number of iterations
max_iter = 100;

% Start EM algorithm
for iter = 1:max_iter
    % E-step
    G = inv(sigma2 * eye(M) + W' * W);
    Ez = G * W' * X';
    Ezz = sigma2 * G + Ez * Ez';

    % M-step
    W_new = X' * Ez' * inv(Ezz);
    sigma2_new = (1 / (N * D)) * (trace(X' * X) - trace(W_new * Ez * X));
%     sigma2_new = (1 / (N * D)) * sum(sum(X.^2 - 2 * X * (W_new * Ez) + (W_new * Ezz) * W_new'));

    % Update parameters
    W = W_new;
    sigma2 = sigma2_new;

    % Calculate log-likelihood
    log_likelihood(iter) = -N*D/2*log(2*pi) - N/2*log(det(sigma2 * eye(M) + W'*W)) - 0.5 * trace(X * inv(sigma2 * eye(M) + W'*W) * X');

    % Check convergence
    if iter > 1 && abs(log_likelihood(iter) - log_likelihood(iter-1)) < 1e-4
        break;
    end
end

% %% EM algorithm for PPCA
% max_iterations = 1000;
% tolerance = 1e-6;
% previous_log_likelihood = -Inf;

% for iter = 1:max_iterations
%      % E-step
%      M_inv = inv(W'*W + sigma2 * eye(M));
%      Ez = M_inv * W' * X'; % MxN
%      Ezz = sigma2 * M_inv + Ez*Ez';
 
%      % M-step
%      W_new = X' * Ez' * inv(Ezz);
%      sigma2_new = (1 / (N * D)) * (trace(X' * X) - trace(X' * Ez' * W_new') ...
%                 - trace(W_new * Ez * X) + trace(W_new * Ezz * W_new'));
 
%      % Update parameters
%      W = W_new;
%      sigma2 = sigma2_new;
 
%      % Compute log likelihood (P(x, z))
%     log_pxz = -(N*D/2) * log(2*pi) - N/2 * log(det(W'*W + sigma2 * eye(M))) ...
%                 - 0.5 * trace(inv(W'*W + sigma2 * eye(M)) * X' * X);

%     % Convergence check
%     if abs(log_pxz - previous_log_likelihood) < tolerance
%         break;
%     end
%     previous_log_likelihood = log_pxz;
% end

%% Plot log likelihood vs iteration
figure;
plot(1:length(log_likelihood), log_likelihood);
xlabel('EM Iteration');
ylabel('Log Data Likelihood');
title('EM Algorithm for PPCA - Log Data Likelihood vs Iteration');

%% Compute PPCA covariance and compare with sample covariance
ppca_covariance = W * W' + sigma2 * eye(D);
sample_covariance = cov(X);

fprintf('PPCA Covariance:\n');
disp(ppca_covariance);
fprintf('Sample Covariance:\n');
disp(sample_covariance);


figure;
hold on;
scatter(Xsim(:,1), Xsim(:,2), 'k');
scatter(mu(1), mu(2), 'g', 'filled', 'SizeData', 100);
pc_start = mu - 2*W';
pc_end = mu + 2*W';
plot([pc_start(1), pc_end(1)], [pc_start(2), pc_end(2)], 'k', 'LineWidth', 2);
Xsim_proj = (W * Ez)' + mu;
scatter(Xsim_proj(:,1), Xsim_proj(:,2), 'r', 'filled');
for i = 1:size(Xsim,1)
    plot([Xsim(i,1), Xsim_proj(i,1)], [Xsim(i,2), Xsim_proj(i,2)], 'r--');
end
axis equal;
legend('Data Points', 'Mean', 'PC Space', 'Projection', 'Projection Error', 'Location', 'Northwest');
title('PCA');
saveas(gcf, 'ps8_2d.png');






