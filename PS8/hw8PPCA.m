clear all; close all;

% Load the data
load('ps8_data.mat');
%% Define parameters
Xsim = Xsim';
X = Xsim;
[D, N] = size(X);
M = 1;
mu = mean(X,2);
X = bsxfun(@minus,X,mu);

% Initialize PPCA parameters
W = rand(D, M);
sigma2 = 1;
log_likelihood = [];

% Define the number of iterations
tol = 1e-4;
max_iter = 500;

% Start EM algorithm
for iter = 1:max_iter
    % E-step
    G = W'*W+sigma2*eye(M);
    U = chol(G);
    WX = W'*X;

    % Calculate log-likelihood
    logdetC = 2*sum(log(diag(U)))+(D-M)*log(sigma2);
    T = U'\WX;
    trInvCS = (dot(X(:),X(:))-dot(T(:),T(:)))/(sigma2*N);
    log_likelihood(iter) = -N*(D*log(2*pi)+logdetC+trInvCS)/2; 

    % Check convergence
    if iter > 1 && abs(log_likelihood(iter) - log_likelihood(iter-1)) < tol
        break;
    end

    Ez = G\WX;   
    V = inv(U);                             
    Ezz = N*sigma2*(V*V')+Ez*Ez';

    % M-step
    U = chol(Ezz);                                           
    W = ((X*Ez')/U)/U';  
    WR = W*U';
    sigma2 = (dot(X(:),X(:))-2*dot(Ez(:),WX(:))+dot(WR(:),WR(:)))/(N*D);

end

%% Plot log likelihood vs iteration
figure;
plot(1:length(log_likelihood), log_likelihood);
xlabel('EM Iteration');
ylabel('Log Data Likelihood');
title('EM Algorithm for PPCA - Log Data Likelihood vs Iteration');
saveas(gcf, 'ps8_2b.png');

%% Compute PPCA covariance and compare with sample covariance
ppca_covariance = W * W' + sigma2 * eye(D);
sample_covariance = cov(X');

fprintf('PPCA Covariance:\n');
disp(ppca_covariance);
fprintf('Sample Covariance:\n');
disp(sample_covariance);


figure;
hold on;
scatter(Xsim(1,:), Xsim(2,:), 'k');
scatter(mu(1), mu(2), 'g', 'filled', 'SizeData', 100);
pc_start = mu - 2*W;
pc_end = mu + 2*W;
plot([pc_start(1), pc_end(1)], [pc_start(2), pc_end(2)], 'k', 'LineWidth', 2);
Xsim_proj = (W * Ez) + mu;
scatter(Xsim_proj(1,:), Xsim_proj(2,:), 'r', 'filled');
for i = 1:size(Xsim,2)
    plot([Xsim(1, i), Xsim_proj(1,i)], [Xsim(2,i), Xsim_proj(2,i)], 'r--');
end
axis equal;
legend('Data Points', 'Mean', 'PC Space', 'Projection', 'Projection Error', 'Location', 'Northwest');
title('PPCA');
saveas(gcf, 'ps8_2d.png');
