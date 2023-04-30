clear all; close all;

% Load the data
load('ps8_data.mat');
Xsim = Xsim';
X = Xsim;
[D, N] = size(X);
M = 1;
mu = mean(X,2);
X = bsxfun(@minus,X,mu);


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
    U = chol(G);
    WInvPsiX = W'*inv(Psi)*X;

    % Calculate log-likelihood
    log_likelihood(iter) = -N*D/2*log(2*pi) - N/2*log(det(Psi + W'*W)) - 0.5 * trace(X' * inv(Psi + W'*W) * X);
    % Check convergence
    if iter > 1 && abs(log_likelihood(iter) - log_likelihood(iter-1)) < 1e-4
        break;
    end

    Ez = G\(W'*inv(Psi)*X);
    V = inv(U);
    Ezz = N*(V*V')+Ez*Ez';
    % M-step
    U = chol(Ezz);
    XEz = X*Ez';
    W = (XEz/U)/U';
    Psi = diag(diag((X*X' - W*Ez*X')/N));
end

% Plot the log data likelihood versus EM iteration
figure;
plot(log_likelihood);
xlabel('EM iteration');
ylabel('Log data likelihood');
title('Log data likelihood vs EM iteration for FA');
saveas(gcf, 'ps8_2e.png')

% FA covariance
FA_cov = W * W' + diag(Psi);

% Check the covariances
sample_cov = cov(X');
disp('Sample covariance:');
disp(sample_cov);
disp('FA covariance:');
disp(FA_cov);

% Plot the data points, mean, low-dimensional space, and projections
figure;
hold on;
scatter(Xsim(1,:), Xsim(2,:), 'k');
scatter(mu(1), mu(2), 'g', 'filled', 'SizeData', 100);
pc_start = mu - 5*W;
pc_end = mu + 5*W;
plot([pc_start(1), pc_end(1)], [pc_start(2), pc_end(2)], 'k', 'LineWidth', 2);
Xsim_proj = (W * Ez) + mu;
scatter(Xsim_proj(1,:), Xsim_proj(2,:), 'r', 'filled');
for i = 1:size(Xsim,2)
    plot([Xsim(1, i), Xsim_proj(1,i)], [Xsim(2,i), Xsim_proj(2,i)], 'r--');
end
axis equal;
legend('Data Points', 'Mean', 'PC Space', 'Projection', 'Projection Error', 'Location', 'Northwest');
title('FA');
saveas(gcf, 'ps8_2g.png')