function [mu, Sigma, pi, gamma] = GMM(Spikes, InitParams1)

% Initialize the model parameters
mu = InitParams1.mu;
K = size(mu, 2);
Sigma = repmat(InitParams1.Sigma, 1, 1, K);
pi = InitParams1.pi;
N = size(Spikes, 2);

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
        Sigma(:, :, k) = (x_minus_mu * diag(gamma(k, :)) * x_minus_mu') / Nk(k);
    end
    pi = Nk / N;

    % Check for convergence
    if iter > 1 && abs(log_likelihood(iter) - log_likelihood(iter-1)) < 1e-10
        log_likelihood = log_likelihood(1:iter);
        break;
    end
end
