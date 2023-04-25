% Load data
load('ps8_data.mat')

% Compute covariance matrix and eigenvalues/vectors
Xplan = Xplan - mean(Xplan, 1);
C = cov(Xplan);
[V, D] = eig(C);

% Sort eigenvectors and eigenvalues in descending order
[lambda, idx] = sort(diag(D), 'descend');
V = V(:, idx);

% (a) Plot eigenvalue spectrum
lambda_sqrt = sqrt(lambda);
plot(lambda_sqrt, 'bo-', 'LineWidth', 2);
xlabel('Index');
ylabel('Square-rooted Eigenvalue');

% Compute variance explained by top 3 principal components
M = 3;
variance_total = sum(lambda);
variance_topM = sum(lambda(1:M));
variance_ratio = variance_topM / variance_total;
fprintf('Percentage of variance explained by top %d PCs: %.2f%%\n', M, variance_ratio * 100);

% (b) Project data into 3D PC space and plot
UM = V(:, 1:M);
Z = Xplan * UM;
figure;
angle_idx = repmat(1:8, [91, 1]);
angle_idx = angle_idx(:);
colors = jet(8);
scatter3(Z(:, 1), Z(:, 2), Z(:, 3), 20, colors(angle_idx, :), 'filled');
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');

% (c) Show UM values using imagesc
figure;
imagesc(UM);
colorbar;
