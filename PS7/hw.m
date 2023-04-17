close all; clear all;
load('ps7_data.mat');

figure;
plot(Spikes, 'k');
xlabel('Time');
ylabel('Voltage (µV)');
title('Voltage vs. Time for Raw Spike Snippets');
saveas(gcf, 'ps7_1a.png'); close all;

%% (b) Apply PCA to all N spike snippets and plot the eigenvector waveforms
% [coeff, score, latent] = pca(Spikes.');
centered_data = Spikes - mean(Spikes, 2);
covariance_matrix = (centered_data * centered_data') / (size(centered_data, 2) - 1);
[coeff, eigenvalues_matrix] = eig(covariance_matrix);
[latent, sort_index] = sort(diag(eigenvalues_matrix), 'descend');
coeff = coeff(:, sort_index);
score = coeff' * centered_data;
score = score';

figure;
plot(coeff(:, 1), 'r');
hold on;
plot(coeff(:, 2), 'g');
plot(coeff(:, 3), 'b');
xlabel('Time');
ylabel('Amplitude (µV)');
title('Eigenvector Waveforms');
legend('1st: red', '2nd: green', '3rd: blue');
saveas(gcf, 'ps7_1b.png'); close all;

%% (c) Plot the square-rooted eigenvalue spectrum
sqrt_eigenvalues = sqrt(latent);
figure;
plot(sqrt_eigenvalues, 'o', "MarkerSize", 5, 'MarkerEdgeColor', 'k');
xlabel('Component Number');
ylabel('Square-rooted Eigenvalue (µV)');
title('Square-rooted Eigenvalue Spectrum');
saveas(gcf, 'ps7_1c.png'); close all;

% Look for an elbow in the eigenvalue spectrum
% (You will need to visually inspect the plot to determine the number of dominant eigenvalues)

%% (d) Create a scatter plot of the PC1 score versus the PC2 score
score(:, 1) = -1 * score(:, 1);
figure;
scatter(score(:, 1), score(:, 2), 5, 'k', 'filled');
xlabel('PC1 Score');
ylabel('PC2 Score');
title('Scatter Plot of PC1 Score vs PC2 Score');
saveas(gcf, 'ps7_1d.png'); close all;

% How many distinct clusters do you see in the plot?
% (You will need to visually inspect the plot to determine the number of clusters)

%% Load your data
% Assuming Spikes is a 31x552 matrix
% Replace 'your_spikes_data.mat' with your actual data file name
% load('your_spikes_data.mat');

%% Cross-validated likelihoods for GMM with K = 1, ..., 8
n_folds = 4;
K_values = 1:8;
n_samples = size(score, 1);
samples_per_fold = n_samples / n_folds;
likelihoods = zeros(1, length(K_values));

for K = K_values
    likelihood = 0;
    for fold = 1:n_folds
        % Create test and train indices for the current fold
        test_indices = (fold - 1) * samples_per_fold + 1 : fold * samples_per_fold;
        train_indices = setdiff(1:n_samples, test_indices);
        
        InitParams_new.mu = InitParams.mu(:, 1:K);
        InitParams_new.Sigma = repmat(InitParams.Sigma, 1, 1, K);
        InitParams_new.pi = ones(1, K) / K; 

        [mu_est, Sigma_est, pi_est] = func_GMM(InitParams_new, score(train_indices, 1:2)');
        
        % Compute likelihood for the test set
        test_data = score(test_indices, 1:2);
        likelihood_fold = 0;
        for n = 1:size(test_data, 1)
            likelihood_n = 0;
            for k = 1:K
                likelihood_n = likelihood_n + pi_est(k) * mvnpdf(test_data(n, :), mu_est(:, k)', Sigma_est(:, :, k));
            end
            likelihood_fold = likelihood_fold + log(likelihood_n);
        end
        likelihood = likelihood + likelihood_fold;
    end
    likelihoods(K) = likelihood;
end

% Plot cross-validated likelihoods versus K
figure;
plot(K_values, likelihoods, 'o-', "MarkerSize", 5, 'MarkerEdgeColor', 'k');
xlabel('K');
ylabel('Cross-validated Likelihood');
title('Cross-validated Likelihoods vs K');
saveas(gcf, 'ps7_2a.png'); close all;

% Optimal value of K
[~, optimal_K] = max(likelihoods);

%% (b) For each value of K = 1, ..., 8, create separate plots
for K = 1:8
    % Train GMM using EM with the first cross-validation fold
    InitParams_new.mu = InitParams.mu(:, 1:K);
    InitParams_new.sigma = repmat(InitParams.Sigma, 1, 1, K);
    InitParams_new.pi = ones(1, K) / K;
    train_indices = samples_per_fold + 1 : n_samples;
    [mu_est, Sigma_est, pi_est] = func_GMM(InitParams_new, score(train_indices, 1:2)');
    
    % Create plot
    figure;
    scatter(score(:, 1), score(:, 2), 5, 'k', 'filled');
    xlabel('PC1 Score');
    ylabel('PC2 Score');
    title(sprintf('Scatter Plot and Ellipses for K = %d', K));
    hold on;
    
    % Plot one-standard-deviation ellipses for each cluster
    for k = 1:K
        func_plotEllipse(mu_est(:, k), Sigma_est(:, :, k));
    end
    hold off;
    saveas(gcf, sprintf('ps7_2b_%d.png', K)); close all;
end

% For K = 3, plot the canonical spike waveform corresponding to each
% cluster center in a “voltage versus time” plot. This will involve projecting the
% two-dimensional µk out into the 31-dimensional space. Use the µk
% from the first
% cross-validation fold.

InitParams_new.mu = InitParams.mu(:, 1:3);
InitParams_new.sigma = repmat(InitParams.Sigma, 1, 1, K);
InitParams_new.pi = ones(1, 3) / 3;
train_indices = samples_per_fold + 1 : n_samples;
[mu_est, Sigma_est, pi_est] = func_GMM(InitParams_new, score(train_indices, 1:2)');

for k = 1:3
    mu_est_31d = coeff(:, 1:2) * mu_est(:, k);
    figure;
    plot(mu_est_31d, "LineWidth", 2.5);
    xlabel('Time');
    ylabel('Voltage (µV)');
    title(sprintf('Canonical Spike Waveform for Cluster %d', k));
    saveas(gcf, sprintf('ps7_2c_%d.png', k)); close all;
end

figure;
hold on;
plot(coeff(:, 1:2) * mu_est(:, 1), 'r');
plot(coeff(:, 1:2) * mu_est(:, 2), 'g');
plot(coeff(:, 1:2) * mu_est(:, 3), 'b');
xlabel('Time');
ylabel('Amplitude (µV)');
title('Eigenvector Waveforms');
legend('1st: red', '2nd: green', '3rd: blue');
saveas(gcf, 'ps7_2c_other.png'); close all;
