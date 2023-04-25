clear all; close all;

% Load the data
load('ps8_data.mat')

% Calculate mean and subtract it from data
mu = mean(Xsim);
Xsim_centered = Xsim - mu;

% Perform PCA
CovMat = cov(Xsim_centered);
[V,D] = eig(CovMat);
[d,ind] = sort(diag(D),'descend');
V = V(:,ind);

% Project data onto PC space
u1 = V(:,1);
Xsim_pc = Xsim_centered * u1;
Xsim_proj = Xsim_centered * u1 * u1' + mu;
% Plot data and projections

figure;
hold on;
scatter(Xsim(:,1), Xsim(:,2), 'k');
scatter(mu(1), mu(2), 'g', 'filled', 'SizeData', 100);
pc_start = mu - 12*u1';
pc_end = mu + 12*u1';
plot([pc_start(1), pc_end(1)], [pc_start(2), pc_end(2)], 'k', 'LineWidth', 2);
scatter(Xsim_proj(:,1), Xsim_proj(:,2), 'r', 'filled');
for i = 1:size(Xsim,1)
    plot([Xsim(i,1), Xsim_proj(i,1)], [Xsim(i,2), Xsim_proj(i,2)], 'r--');
end
axis equal;
legend('Data Points', 'Mean', 'PC Space', 'Projection', 'Projection Error', 'Location', 'Northwest');
title('PCA');
saveas(gcf, 'ps8_2a.png');