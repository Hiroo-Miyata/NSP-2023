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
saveas(gcf, 'ps5_2.png');

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
