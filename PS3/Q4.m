close all; clear all;
load("ps3_realdata.mat");

% The .mat file has two variables: train_trial
% contains the training data and test_trial contains the test data. Each variable is
% a structure of dimensions (91 trials) × (8 reaching angles). Each structure contains
% spike trains recorded simultaneously from 97 neurons while the monkey reached 91
% times along each of 8 different reaching angles.
% The spike train recorded from the ith neuron on the nth trial of the kth reaching angle
% is contained in train_trial(n,k).spikes(i,:), where n = 1, . . . , 91, k = 1, . . . , 8,
% and i = 1, . . . , 97. A spike train is represented as a sequence of zeros and ones, where
% time is discretized in 1 ms steps. A zero indicates that the neuron did not spike in the
% 1 ms bin, whereas a one indicates that the neuron spiked once in the 1 ms bin. The
% structure test_trial has the same format as train_trial

% These spike trains were recorded while the monkey performed a delayed reaching task,
% as described in class. Each spike train is 700 ms long (and is thus represented by a
% 1 × 700 vector), which comprises a 200 ms baseline period (before the reach target
% turned on) and a 500 ms planning period (after the reach target turned on). 
% Because it takes time for information about the reach target to arrive in premotor cortex
% (due to the time required for action potentials to propagage and for visual processing), 
% we will ignore the first 150 ms of the planning period. For this problem, we
% will take spike counts for each neuron within a single 200 ms bin starting 150 ms 
% after the reach target turns on. In other words, we will only use
% train_trial(n,k).spikes(i,351:550) and test_trial(n,k).spikes(i,351:550)
% in this problem.

ntrain = size(train_trial,1);
ntest = size(test_trial,1);
nclasses = size(train_trial,2);
nneurons = size(train_trial(1,1).spikes,1);
timeWindow = 351:550;
timebin = 200;

% a. we assume that each class is gaussian distributed with a shared covariance matrix
% Fit the ML parameters of the model to the training data (91 × 8
% data points). Then, use these parameters to classify the test data (91 × 8 data
% points) by finding the k which maximize p(x|C_k). What is the percent of test data points
% correctly classified?

% 1. Compute the mean and covariance matrix for each class

mu = zeros(nneurons, nclasses);
sigma = zeros(nneurons, nneurons);
for k = 1:nclasses
    % compute the mean
    spike_train = cat(3, train_trial(:,k).spikes); % 97 x 700 x 91
    spike_counts = squeeze(sum(spike_train(:, timeWindow, :),2));
    mu(:,k) = squeeze(mean(spike_counts,2));
    % compute the covariance matrix
    sigma = sigma + cov(spike_counts');
end
sigma = sigma/nclasses;

% 2. Compute the probability of each class for each test data point
% p(x|C_k) = N(x|mu_k, sigma)
% p(C_k) = 1/8

p_x_k = @(x, k) mu(:,k)' * inv(sigma) * x - 0.5 * mu(:,k)' * inv(sigma) * mu(:,k);
argmaxk = @(x) find([p_x_k(x,1) p_x_k(x,2) p_x_k(x,3) p_x_k(x,4) p_x_k(x,5) p_x_k(x,6) p_x_k(x,7) p_x_k(x,8)] == max([p_x_k(x,1) p_x_k(x,2) p_x_k(x,3) p_x_k(x,4) p_x_k(x,5) p_x_k(x,6) p_x_k(x,7) p_x_k(x,8)]));

% 3. Compute the percent of test data points correctly classified
correct = 0;
for n = 1:ntest
    for k = 1:nclasses
        spike_count = squeeze(sum(test_trial(n,k).spikes(:,timeWindow),2));
        if argmaxk(spike_count) == k
            correct = correct + 1;
        end
    end
end
fprintf('Percent of test data points correctly classified: %f \n', correct/(ntest*nclasses));
% Percent of test data points correctly classified: 0.960165 

% % b.  Repeat part (a) for model which is gaussian distribusion
% % with individual covariance matrix. You should encounter a Matlab
% % warning when classifying the test data. Why did the Matlab warning occur?
% 
% % 1. Compute the mean and covariance matrix for each class
% 
% mu = zeros(nneurons, nclasses);
% sigma = zeros(nneurons, nneurons, nclasses);
% for k = 1:nclasses
%     % compute the mean
%     spike_train = cat(3, train_trial(:,k).spikes); % 97 x 700 x 91
%     spike_counts = squeeze(sum(spike_train(:, timeWindow, :),2));
%     mu(:,k) = squeeze(mean(spike_counts,2));
%     % compute the covariance matrix
%     sigma(:,:,k) = cov(spike_counts');
% end
% 
% % 2. Compute the probability of each class for each test data point
% % p(x|C_k) = N(x|mu_k, sigma_k)
% % p(C_k) = 1/8
% 
% p_x_k = @(x, k) - (x - mu(:,k))' * inv(squeeze(sigma(:,:,k))) * (x - mu(:,k)) - log(det(squeeze(sigma(:,:,k))));
% argmaxk = @(x) find([p_x_k(x,1) p_x_k(x,2) p_x_k(x,3) p_x_k(x,4) p_x_k(x,5) p_x_k(x,6) p_x_k(x,7) p_x_k(x,8)] == max([p_x_k(x,1) p_x_k(x,2) p_x_k(x,3) p_x_k(x,4) p_x_k(x,5) p_x_k(x,6) p_x_k(x,7) p_x_k(x,8)]));
% 
% % 3. Compute the percent of test data points correctly classified
% correct = 0;
% for n = 1:ntest
%     for k = 1:nclasses
%         spike_count = squeeze(sum(test_trial(n,k).spikes(:,timeWindow),2));
%         if argmaxk(spike_count) == k
%             correct = correct + 1;
%         end
%     end
% end
% fprintf('Percent of test data points correctly classified: %f \n', correct/(ntest*nclasses));
% 
% % the reason of waring is that the covariance matrix can not be inverted


% c. use naive bayes classifier to classify the test data. What is the percent of test data points correctly classified?

% 1. Compute the lambda for each class

lambda = zeros(nneurons, nclasses);
for k = 1:nclasses
    % compute the mean
    spike_train = cat(3, train_trial(:,k).spikes); % 97 x 700 x 91
    spike_counts = squeeze(sum(spike_train(:, timeWindow, :),2));
    lambda(:,k) = squeeze(mean(spike_counts,2));
end

probability = zeros(nclasses, 1);

% 2. Compute the probability of each class for each test data point

p_x_k = @(x, k) prod(poisspdf(x, lambda(:,k)));
argmaxk = @(x) find([p_x_k(x,1) p_x_k(x,2) p_x_k(x,3) p_x_k(x,4) p_x_k(x,5) p_x_k(x,6) p_x_k(x,7) p_x_k(x,8)] == max([p_x_k(x,1) p_x_k(x,2) p_x_k(x,3) p_x_k(x,4) p_x_k(x,5) p_x_k(x,6) p_x_k(x,7) p_x_k(x,8)]));

% 3. Compute the percent of test data points correctly classified

correct = 0;
for n = 1:ntest
    for k = 1:nclasses
        spike_count = squeeze(sum(test_trial(n,k).spikes(:,timeWindow),2));
        if argmaxk(spike_count) == k
            correct = correct + 1;
        end
    end
end
fprintf('Percent of test data points correctly classified: %f \n', correct/(ntest*nclasses));
