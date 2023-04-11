clear all; close all;

load("ps6_data.mat");

% Initialize the model parameters
K = 3;
mu = InitParams1.mu;
Sigma = repmat(InitParams1.Sigma, 1, 1, K);
pi = InitParams1.pi;
N = size(Spikes, 2);

% Initialize the model parameters
% K = 3;
% mu = InitParams2.mu;
% Sigma = repmat(InitParams2.Sigma, 1, 1, K);
% pi = InitParams2.pi;
% N = size(Spikes, 2);

% EM algorithm
maxIter = 100;
log_likelihood = zeros(maxIter, 1);

for iter = 1:maxIter
    % E-step
    log_gamma = zeros(K, N);
    for k = 1:K
        log_gamma(k, :) = log(pi(k)) + logmvnpdf(Spikes', mu(:, k)', Sigma(:, :, k));
    end
    logsumexp_gamma = logsumexp(log_gamma, 1);
    log_likelihood(iter) = sum(logsumexp_gamma);
    gamma = exp(log_gamma - logsumexp_gamma);
    
    % M-step 
    Nk = sum(gamma, 2);
    for k = 1:K
        mu(:, k) = (Spikes * gamma(k, :)') / Nk(k);
        x_minus_mu = Spikes - mu(:, k);
        Sigma = zeros(31, 31, K);
        for i = 1:N
            Sigma(:, :, k) = Sigma(:, :, k) + gamma(k, i) * (x_minus_mu(:, i) * x_minus_mu(:, i)');
        end
        Sigma(:, :, k) = Sigma(:, :, k) / Nk(k);

        % Check if the covariance matrix is well-conditioned and invertible
        cond_number = cond(Sigma(:, :, k));
        determinant = det(Sigma(:, :, k));
        if cond_number >= 1e10 || determinant <= 1e-10
            fprintf('Warning: Ill-conditioned covariance matrix at iteration %d, cluster %d\n', iter, k);
            fprintf('Condition number: %g, determinant: %g\n', cond_number, determinant);
        end
    end
    pi = Nk / N;

    % Check for convergence
    if iter > 1 && abs(log_likelihood(iter) - log_likelihood(iter-1)) < 1e-10
        log_likelihood = log_likelihood(1:iter);
        break;
    end
end

% Plot log likelihood
figure;
plot(1:length(log_likelihood), log_likelihood);
xlabel('EM Iteration Number');
ylabel('Log Likelihood');
title('Log Likelihood vs. EM Iteration Number');
saveas(gcf, 'log_likelihood.png');

% Store estimates
mu_k = mu;
sigma_k = Sigma;
pi_k = pi;

% Display pi_k
disp('pi_k:');
disp(pi_k);

% Plot voltage vs time for each cluster
for k = 1:K
    figure;
    hold on;
    
    % Plot waveform snippets assigned to the kth neuron
    [~, cluster_assignments] = max(gamma, [], 1);
    plot(Spikes(:, cluster_assignments == k), 'Color', [0.8 0.8 0.8]);
    
    % Plot cluster center and standard deviation bounds
    plot(mu_k(:, k), 'r', 'LineWidth', 2);
    plot(mu_k(:, k) + sqrt(diag(sigma_k(:, :, k))), 'r--');
    plot(mu_k(:, k) - sqrt(diag(sigma_k(:, :, k))), 'r--');
    
    xlabel('Time');
    ylabel('Voltage (\muV)');
    title(['Cluster ' num2str(k)]);
    ylim([-600 1200])
    hold off;
    saveas(gcf, ['cluster_' num2str(k) '.png']);
end



% initial_covariance = InitParams1.Sigma;

% % Check the condition number
% cond_number = cond(initial_covariance);
% fprintf('Condition number: %g\n', cond_number);

% % Check the determinant
% determinant = det(initial_covariance);
% fprintf('Determinant: %g\n', determinant);

% % Verify if the matrix is well-conditioned and invertible
% if cond_number < 1e10 && determinant > 1e-10
%     disp('The initial covariance matrix is well-conditioned and invertible.');
% else
%     disp('The initial covariance matrix is ill-conditioned or not invertible.');
% end

% initial_covariance = InitParams2.Sigma;

% % Check the condition number
% cond_number = cond(initial_covariance);
% fprintf('Condition number: %g\n', cond_number);

% % Check the determinant
% determinant = det(initial_covariance);
% fprintf('Determinant: %g\n', determinant);

% % Verify if the matrix is well-conditioned and invertible
% if cond_number < 1e10 && determinant > 1e-10
%     disp('The initial covariance matrix is well-conditioned and invertible.');
% else
%     disp('The initial covariance matrix is ill-conditioned or not invertible.');
% end
