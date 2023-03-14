clear all; close all;

% We will analyze real neural data recorded using a 100-electrode array in premotor cortex of a macaque monkey
% The dataset is ps2_data.mat

% The following describes the data format. The .mat file has a single variable named
% trial, which is a structure of dimensions (182 trials) × (8 reaching angles). The
% structure contains spike trains recorded from a single neuron while the monkey reached
% 182 times along each of 8 different reaching angles (where the trials of different reaching
% angles were interleaved). The spike train for the nth trial of the kth reaching angle is
% contained in trial(n,k).spikes, where n = 1, . . . , 182 and k = 1, . . . , 8. The indices
% k = 1,...,8 correspond to reaching angles 30/180 * pi, 70/180 * pi, 110/180 * pi, 150/180 * pi, 190/180 * pi, 230/180 * pi, 310/180 * pi, 350/180 * pi.
% The reaching angles are not evenly spaced around the circle due to
% experimental constraints that are beyond the scope of this problem set.

load('ps2_data.mat');
ntrials = 182;
nangles = 8;
nneurons = 100;
angles = [30/180 * pi, 70/180 * pi, 110/180 * pi, 150/180 * pi, 190/180 * pi, 230/180 * pi, 310/180 * pi, 350/180 * pi];

positions = [6, 3, 2, 1, 4, 7, 8, 9];

% A spike train is represented as a sequence of zeros and ones, where time is discretized
% in 1 ms steps. A zero indicates that the neuron did not spike in the 1 ms bin, whereas
% a one indicates that the neuron spiked once in the 1 ms bin. Due to the refractory
% period, it is not possible for a neuron to spike more than once within a 1 ms bin. Each
% spike train is 500 ms long and is, thus, represented by a 1 × 500 vector.
% the center of the figure is arrows-Q4.png

% (a) Plot 5 spike trains for each reaching angle

figure;
for i = 1:nangles
    subplot(3, 3, positions(i));
    for j = 1:5
        X = find(trial(j, i).spikes);
        Y = ones(1, length(X));
        scatter(X, j*Y, 'k.');
        hold on;
    end
    % remove the ticks
    set(gca, 'YTick', []);
    ylim([0.5 5.5]);

    if i == 8
        xlabel("Time (msec)")
    end
end
subplot(3, 3, 5);
axis off;
I = imread('arrows-Q4.png', 'BackgroundColor', [1 1 1]);
imshow(I);

saveas(gcf, 'results/4-a.png'); close all;


% (b) For each reaching angle, find the spike histogram by taking spike counts in non-overlapping 20 ms bins.
% Then averaging across the 182 trials. The spike histograms
% should have firing rate (in spikes / second) as the vertical axis and time (in msec,
% not time bin index) as the horizontal axis. Plot the 8 resulting spike histograms
% around a circle, as in part (a).

figure;
spikeCounts20ms = zeros(ntrials, nangles, 500/20);
for i = 1:nangles
    subplot(3,3, positions(i));
    for j = 1:ntrials
        spikeTrain = trial(j, i).spikes;
        spikeCounts20ms(j, i, :) = sum(reshape(spikeTrain, 20, 25), 1);
    end
    meanSpikeCounts20ms = mean(spikeCounts20ms(:, i, :), 1);
    meanSpikeCounts20ms = squeeze(meanSpikeCounts20ms) * 1000 / 20;
    bar(10:20:500, meanSpikeCounts20ms, 1);
    % plot(10:20:500, meanSpikeCounts20ms, "k", "LineWidth", 1.5);
    hold on;

    if i == 8
        ylabel("f (Hz)")
        xlabel("Time (msec)")
    end
end
subplot(3, 3, 5);
axis off;
I = imread('arrows-Q4.png', 'BackgroundColor', [1 1 1]);
imshow(I);
saveas(gcf, 'results/4-b.png'); close all;

% (c) For each trial, count the number of spikes across the entire trial. Plots these
% points on the axes below
% x-axis: reaching angle (in degree)
% y-axis: firing rate (in spikes / second)
% There should be 182 ∗ 8 points
% in the plot (but some points may be on top of each other due to the discrete
% nature of spike counts). For each reaching angle, find the mean firing rate across
% the 182 trials, and plot the mean firing rate using a red point on the same plot.
% Then, fit the cosine tuning curve (1) to the 8 red points by minimizing the sum of squared errors
% the sum of square error is
% sum((lambda(s_i) -  r_0 - (r_max - r_0) * cos(s_i - s_max))^2)
% with respect to the parameters r_0, r_max, and s_max. Plot the cosine tuning curve of this neuron in green on the same plot
% the parameters is done using linear regression

figure;
spikeCounts = zeros(ntrials, nangles);
meanFireRates = zeros(1, 8);
for i = 1:nangles
    for j = 1:ntrials
        spikeTrain = trial(j, i).spikes;
        spikeCounts(j, i) = sum(spikeTrain);
        % plot all spike counts in one figure (x-axis: reaching angle, y-axis: spike counts)
        % opacity: 0.1
        scatter(angles(i) * 180 / pi, spikeCounts(j, i) * 2, 50, 'blue', 'filled', 'MarkerFaceAlpha', 0.1); hold on;
    end
    meanFireRates(i) = mean(spikeCounts(:, i)) / 0.5;
end

scatter(angles * 180 / pi, meanFireRates, 50, 'red', 'filled'); hold on;
SEs = @(x) sum((meanFireRates - x(1) - (x(2) - x(1)) * cos(angles - x(3))).^2);
x0 = [0,1,0];
theta = fminsearch(SEs, x0);
r_0 = theta(1);
r_max = theta(2);
s_max = theta(3);
plot(angles * 180 / pi, r_0 + (r_max - r_0) * cos(angles - s_max), 'g', "LineWidth", 2);

xlim([0 360])
xlabel("S (movement direction in degrees")
ylabel("f (Hz)")
saveas(gcf, 'results/4-c.png'); close all;


% (d) For each reaching angle, plot the normalized distribution of spike counts (using
% the same counts from part (c)). Plot the 8 distributions around a circle, as in part
% (a). Fit a Poisson distribution to each empirical distribution and plot it on top of
% the corresponding empirical distribution.

figure;
for i = 1:nangles
    subplot(3, 3, positions(i));
    histogram(spikeCounts(:, i), 'Normalization', 'probability');
    hold on;
    x = 0:1:max(spikeCounts(:, i));
    lambda = mean(spikeCounts(:, i));
    plot(x, poisspdf(x, lambda), 'r', 'LineWidth', 2);

    if i == 8
        xlabel("Spike Counts")
        ylabel("Probability")
    end

end
subplot(3, 3, 5);
axis off;
I = imread('arrows-Q4.png', 'BackgroundColor', [1 1 1]);
imshow(I);
saveas(gcf, 'results/4-d.png'); close all;

% (e) For each reaching angle, find the mean and variance of the spike counts across the
% 182 trials (using the same spike counts from part (c)). Plot the obtained mean
% and variance on x and y axes, respectively. There should be 8 points in the plot

figure;
meanSpikeCounts = zeros(1, 8);
varSpikeCounts = zeros(1, 8);
for i = 1:nangles
    meanSpikeCounts(i) = mean(spikeCounts(:, i));
    varSpikeCounts(i) = var(spikeCounts(:, i));
end
scatter(meanSpikeCounts, varSpikeCounts, 50, "k", "filled");
hold on;
xlabel("mean (spikes)")
ylabel("variance (spikes^2)")
axis equal
% add x = y line
x = 0:1:max(meanSpikeCounts);
plot(x, x, 'r', 'LineWidth', 2);

saveas(gcf, 'results/4-e.png'); close all;

% (f) For each reaching angle, plot the normalized distribution of ISIs.
% Plot the 8 distributions around a circle, as in part (a). Fit an exponential
% distribution to each empirical distribution and plot it on top of the corresponding
% empirical distribution.

figure;
for i = 1:nangles
    subplot(3, 3, positions(i));
    ISIs = [];
    for j = 1:ntrials
        ISIs = [ISIs, diff(find(trial(j, i).spikes))];
    end
    ISIs = ISIs / 1000;
    histogram(ISIs, 'Normalization', 'pdf')
    hold on
    x = 0:0.001:max(ISIs);
    y = exppdf(x, 1 / mean(spikeCounts(:, i)));
    plot(x, y, 'r', 'LineWidth', 2)

    if i == 8
        xlabel("ISIs (s)")
        ylabel("Probability Density")
    end
end
subplot(3, 3, 5);
axis off;
I = imread('arrows-Q4.png', 'BackgroundColor', [1 1 1]);
imshow(I);
saveas(gcf, 'results/4-f.png'); close all;