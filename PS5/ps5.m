close all;
load("ps5_data.mat");
% in the data file, the following variables are defined:
% RealWaveform: 10 seconds of real electrode data sampled at 30 kHz. Values are in µV.
% InitTwoClusters_1, InitTwoClusters_2, InitThreeClusters_1, InitThreeClusters_2:
% each is a 31 x K matrix, where the kth column is the initialization of the kth cluster center, µk (k = 1, . . . , K).


% Problem 1: High-pass filtering

x = RealWaveform;
f_0 = 30000; % sampling rate of waveform (Hz)
f_stop = 250; % stop frequency (Hz)
f_Nyquist = f_0/2; % Nyquist frequency (Hz)
n = length(x); % number of samples
f_all = linspace(-f_Nyquist, f_Nyquist, n); % frequency vector
desired_response = ones(n,1); % desired response
desired_response(abs(f_all) <= f_stop) = 0; % set desired response to 0 for frequencies above f_stop
x_filtered = real(ifft(fft(x).*fftshift(desired_response))); % filter the signal


% Problem 2: Spike detection
% Set a threshold, Vthresh = 250 µV, and determine all of the times that the signal crosses
% from below to above the threshold. Plot the threshold as a line across your plot of the
% high-pass filtered waveform from Problem 1.
% Take 1 ms snippets of the waveform beginning 0.3 ms before each threshold crossing.
% Each snippet should have 31 samples; the tenth sample should be less than Vthresh, and
% the eleventh sample should be greater than Vthresh. In a new figure, make a “voltage
% versus time” plot containing the following:
% 1. all of the threshold-crossing waveform snippets, and
% 2. the threshold as a horizontal line.
% Your result should look similar to Figure 3 from Lewicki’s review paper, with the key
% difference that your plot should show multiple spike shapes (instead of just one). You
% will also notice a few non-stereotyped traces resulting from either different neurons
% spiking near the same time or noise exceeding threshold

Vthresh = 250; % threshold (µV)
crossings = find(x_filtered(1:end-1) < Vthresh & x_filtered(2:end) >= Vthresh); % find threshold crossings
crossings = crossings + 1; % adjust for indexing
crossings = round(crossings); % round to nearest integer

snippets = zeros(31, length(crossings)); % initialize snippets matrix
for i = 1:length(crossings)
    snippets(:,i) = x_filtered(crossings(i)-10:crossings(i)+20); % get snippets
end

figure;
plot(snippets, "Color", "k"); % plot snippets
hold on;
plot([1 31], [Vthresh Vthresh], 'r', 'LineWidth', 1.5); % plot threshold
xticks([1 10 19 31]);
xticklabels([-0.3 0 0.3 0.7]);
xlim([1 31]);
xlabel('Time (ms)');
ylabel('Voltage (µV)');
title('Voltage vs. Time Plot');
saveas(gcf, 'ps5_2.png'); close all;

% Problem 3: Clustering with the K-mean algorithm
% Implement the K-means algorithm in MATLAB, and use it to determine the neuron
% responsible for each recorded spike.
% Treat each snippet as a point x_n ∈ R^D (n = 1, ..., N), where D = 31 is the number of
% samples in each snippet, and N is the number of detected spikes. In this problem, we
% will assume that there are K = 2 neurons contributing spikes to the recorded waveform.
% Initialize the cluster centers using InitTwoClusters 1

% (a) For each cluster (k = 1, 2), create a separate “voltage versus time” plot containing the following:
% 1. the cluster center µk returned by K-means as a red waveform trace (i.e., the prototypical action potential for the kth neuron),
% 2. all of the waveform snippets assigned to the kth neuron, and
% 3. the threshold as a horizontal line.

K = 2; % number of clusters
D = 31; % number of samples in each snippet
N = length(crossings); % number of detected spikes
x = snippets; % snippets matrix

% initialize cluster centers
mu = zeros(D, K);
mu(:,1) = InitTwoClusters_1(:,1);
mu(:,2) = InitTwoClusters_1(:,2);

% classify snippets
r = zeros(N, K);
for i = 1:N
    r(i, 1) = norm(x(:,i) - mu(:,1));
    r(i, 2) = norm(x(:,i) - mu(:,2));
end

label = zeros(N, 1);
for i = 1:N
    [~, label(i)] = min(r(i,:));
end

% plot
figure;
plot(x(:,label==1), "Color", "k"); % plot snippets
hold on;
plot(mu(:,1), 'r', 'LineWidth', 1.5); % plot cluster center
plot([1 31], [Vthresh Vthresh], 'r', 'LineWidth', 1.5); % plot threshold
xticks([1 10 19 31]);
xticklabels([-0.3 0 0.3 0.7]);
xlim([1 31]);
xlabel('Time (ms)');
ylabel('Voltage (µV)');
title('Voltage vs. Time Plot (Cluster 1)');
saveas(gcf, 'ps5_3a1.png');

figure;
plot(x(:,label==2), "Color", "k"); % plot snippets
hold on;
plot(mu(:,2), 'r', 'LineWidth', 1.5); % plot cluster center
plot([1 31], [Vthresh Vthresh], 'r', 'LineWidth', 1.5); % plot threshold
xticks([1 10 19 31]);
xticklabels([-0.3 0 0.3 0.7]);
xlim([1 31]);
xlabel('Time (ms)');
ylabel('Voltage (µV)');
title('Voltage vs. Time Plot (Cluster 2)');
saveas(gcf, 'ps5_3a2.png'); close all;


% (b) Plot the objective function J versus iteration number. How many iterations did it take for K-means to converge?

J = zeros(1, 100); % initialize objective function

for iter = 1:100
    % update cluster centers
    for k = 1:K
        mu(:,k) = mean(x(:,label==k), 2);
    end
    
    % update labels
    for i = 1:N
        r(i, 1) = norm(x(:,i) - mu(:,1));
        r(i, 2) = norm(x(:,i) - mu(:,2));
    end

    for i = 1:N
        [~, label(i)] = min(r(i,:));
    end
    
    % compute objective function
    J(iter) = 0;
    for i = 1:N
        J(iter) = J(iter) + norm(x(:,i) - mu(:,label(i)))^2;
    end
end

figure;
plot(J, 'k', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Objective Function');
title('Objective Function vs. Iteration');
saveas(gcf, 'ps5_3b.png'); close all;


% Problem 4. As discussed in class, K-means guarantees convergence to a local optimum.
% Thus, it is possible to converge to different local optima with different initializations.
% Repeat Problem 3 where the cluster centers are initialized using InitTwoClusters 2.
% Is the local optimum found here the same or different as that found in Problem 3?

% initialize cluster centers
mu2 = zeros(D, K);
mu2(:,1) = InitTwoClusters_2(:,1);
mu2(:,2) = InitTwoClusters_2(:,2);

% classify snippets
r = zeros(N, K);
for i = 1:N
    r(i, 1) = norm(x(:,i) - mu2(:,1));
    r(i, 2) = norm(x(:,i) - mu2(:,2));
end

label = zeros(N, 1);
for i = 1:N
    [~, label(i)] = min(r(i,:));
end

% plot
figure;
plot(x(:,label==1), "Color", "k"); % plot snippets
hold on;
plot(mu2(:,1), 'r', 'LineWidth', 1.5); % plot cluster center
plot([1 31], [Vthresh Vthresh], 'r', 'LineWidth', 1.5); % plot threshold
xticks([1 10 19 31]);
xticklabels([-0.3 0 0.3 0.7]);
xlim([1 31]);
xlabel('Time (ms)');
ylabel('Voltage (µV)');
title('Voltage vs. Time Plot (Cluster 1)');
saveas(gcf, 'ps5_4a1.png');

figure;
plot(x(:,label==2), "Color", "k"); % plot snippets
hold on;
plot(mu2(:,2), 'r', 'LineWidth', 1.5); % plot cluster center
plot([1 31], [Vthresh Vthresh], 'r', 'LineWidth', 1.5); % plot threshold
xticks([1 10 19 31]);
xticklabels([-0.3 0 0.3 0.7]);
xlim([1 31]);
xlabel('Time (ms)');
ylabel('Voltage (µV)');
title('Voltage vs. Time Plot (Cluster 2)');
saveas(gcf, 'ps5_4a2.png'); close all;

% plot objective function

J = zeros(1, 100); % initialize objective function

for iter = 1:100
    % update cluster centers
    for k = 1:K
        mu2(:,k) = mean(x(:,label==k), 2);
    end
    
    % update labels
    for i = 1:N
        r(i, 1) = norm(x(:,i) - mu2(:,1));
        r(i, 2) = norm(x(:,i) - mu2(:,2));
    end

    for i = 1:N
        [~, label(i)] = min(r(i,:));
    end
    
    % compute objective function
    J(iter) = 0;
    for i = 1:N
        J(iter) = J(iter) + norm(x(:,i) - mu2(:,label(i)))^2;
    end
end

figure;
plot(J, 'k', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Objective Function');
title('Objective Function vs. Iteration');
saveas(gcf, 'ps5_4b.png'); close all;

% Problem 5 (a) Repeat Problem 3 with K = 3 and initializing using InitThreeClusters_1.
% How does the local optimum found here differ from that found in Problem 3?

% initialize cluster centers
K = 3;
mu3 = zeros(D, K);
mu3(:,1) = InitThreeClusters_1(:,1);
mu3(:,2) = InitThreeClusters_1(:,2);
mu3(:,3) = InitThreeClusters_1(:,3);

% classify snippets
r = zeros(N, K);
for i = 1:N
    r(i, 1) = norm(x(:,i) - mu3(:,1));
    r(i, 2) = norm(x(:,i) - mu3(:,2));
    r(i, 3) = norm(x(:,i) - mu3(:,3));
end

label = zeros(N, 1);
for i = 1:N
    [~, label(i)] = min(r(i,:));
end

% plot
figure;
plot(x(:,label==1), "Color", "k"); % plot snippets
hold on;
plot(mu3(:,1), 'r', 'LineWidth', 1.5); % plot cluster center
plot([1 31], [Vthresh Vthresh], 'r', 'LineWidth', 1.5); % plot threshold
xticks([1 10 19 31]);
xticklabels([-0.3 0 0.3 0.7]);
xlim([1 31]);
xlabel('Time (ms)');
ylabel('Voltage (µV)');
title('Voltage vs. Time Plot (Cluster 1)');
saveas(gcf, 'ps5_5a1.png');

figure;
plot(x(:,label==2), "Color", "k"); % plot snippets
hold on;
plot(mu3(:,2), 'r', 'LineWidth', 1.5); % plot cluster center
plot([1 31], [Vthresh Vthresh], 'r', 'LineWidth', 1.5); % plot threshold
xticks([1 10 19 31]);
xticklabels([-0.3 0 0.3 0.7]);
xlim([1 31]);
xlabel('Time (ms)');
ylabel('Voltage (µV)');
title('Voltage vs. Time Plot (Cluster 2)');
saveas(gcf, 'ps5_5a2.png');

figure;
plot(x(:,label==3), "Color", "k"); % plot snippets
hold on;
plot(mu3(:,3), 'r', 'LineWidth', 1.5); % plot cluster center
plot([1 31], [Vthresh Vthresh], 'r', 'LineWidth', 1.5); % plot threshold
xticks([1 10 19 31]);
xticklabels([-0.3 0 0.3 0.7]);
xlim([1 31]);
xlabel('Time (ms)');
ylabel('Voltage (µV)');
title('Voltage vs. Time Plot (Cluster 3)');
saveas(gcf, 'ps5_5a3.png'); close all;

% plot objective function

J = zeros(1, 100); % initialize objective function

for iter = 1:100
    % update cluster centers
    for k = 1:K
        mu3(:,k) = mean(x(:,label==k), 2);
    end
    
    % update labels
    for i = 1:N
        r(i, 1) = norm(x(:,i) - mu3(:,1));
        r(i, 2) = norm(x(:,i) - mu3(:,2));
        r(i, 3) = norm(x(:,i) - mu3(:,3));
    end

    for i = 1:N
        [~, label(i)] = min(r(i,:));
    end
    
    % compute objective function
    J(iter) = 0;
    for i = 1:N
        J(iter) = J(iter) + norm(x(:,i) - mu3(:,label(i)))^2;
    end
end

figure;
plot(J, 'k', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Objective Function');
title('Objective Function vs. Iteration');
saveas(gcf, 'ps5_5b.png'); close all;


% Problem 5 (b) Repeat Problem 3 with K = 3 and initializing using InitThreeClusters_2.
% Is the local optimum found here the same or different as that found in part (a)?

% initialize cluster centers
K = 3;
mu4 = zeros(D, K);
mu4(:,1) = InitThreeClusters_2(:,1);
mu4(:,2) = InitThreeClusters_2(:,2);
mu4(:,3) = InitThreeClusters_2(:,3);

% classify snippets
r = zeros(N, K);
for i = 1:N
    r(i, 1) = norm(x(:,i) - mu4(:,1));
    r(i, 2) = norm(x(:,i) - mu4(:,2));
    r(i, 3) = norm(x(:,i) - mu4(:,3));
end

label = zeros(N, 1);
for i = 1:N
    [~, label(i)] = min(r(i,:));
end

% plot
figure;
plot(x(:,label==1), "Color", "k"); % plot snippets
hold on;
plot(mu4(:,1), 'r', 'LineWidth', 1.5); % plot cluster center
plot([1 31], [Vthresh Vthresh], 'r', 'LineWidth', 1.5); % plot threshold
xticks([1 10 19 31]);
xticklabels([-0.3 0 0.3 0.7]);
xlim([1 31]);
xlabel('Time (ms)');
ylabel('Voltage (µV)');
title('Voltage vs. Time Plot (Cluster 1)');
saveas(gcf, 'ps5_5c1.png');

figure;
plot(x(:,label==2), "Color", "k"); % plot snippets
hold on;
plot(mu4(:,2), 'r', 'LineWidth', 1.5); % plot cluster center
plot([1 31], [Vthresh Vthresh], 'r', 'LineWidth', 1.5); % plot threshold
xticks([1 10 19 31]);
xticklabels([-0.3 0 0.3 0.7]);
xlim([1 31]);
xlabel('Time (ms)');
ylabel('Voltage (µV)');
title('Voltage vs. Time Plot (Cluster 2)');
saveas(gcf, 'ps5_5c2.png');

figure;
plot(x(:,label==3), "Color", "k"); % plot snippets
hold on;
plot(mu4(:,3), 'r', 'LineWidth', 1.5); % plot cluster center
plot([1 31], [Vthresh Vthresh], 'r', 'LineWidth', 1.5); % plot threshold
xticks([1 10 19 31]);
xticklabels([-0.3 0 0.3 0.7]);
xlim([1 31]);
xlabel('Time (ms)');
ylabel('Voltage (µV)');
title('Voltage vs. Time Plot (Cluster 3)');
saveas(gcf, 'ps5_5c3.png'); close all;

% plot objective function

J = zeros(1, 100); % initialize objective function

for iter = 1:100
    % update cluster centers
    for k = 1:K
        mu4(:,k) = mean(x(:,label==k), 2);
    end
    
    % update labels
    for i = 1:N
        r(i, 1) = norm(x(:,i) - mu4(:,1));
        r(i, 2) = norm(x(:,i) - mu4(:,2));
        r(i, 3) = norm(x(:,i) - mu4(:,3));
    end

    for i = 1:N
        [~, label(i)] = min(r(i,:));
    end
    
    % compute objective function
    J(iter) = 0;
    for i = 1:N
        J(iter) = J(iter) + norm(x(:,i) - mu4(:,label(i)))^2;
    end
end

figure;
plot(J, 'k', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Objective Function');
title('Objective Function vs. Iteration');
saveas(gcf, 'ps5_5d.png'); close all;


