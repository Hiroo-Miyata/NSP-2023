clear all; close all;
% we will consider a simulated neuron that has a cosine tuning curve described in equation 1
% equation 1: f(s) = r_0 + (r_max - r_0) * cos(s - s_max)
% where f is the firing rate (in spikes per second), s is the reaching angle of the arm
% s_max is the reaching angle associated with the maximum response r_max, and r_0 is the baseline firing rate
% and r_0 is an offset that shifts the tuning curve up from the zero axis.
% Let r_0 = 30, r_max = 50, and s_max = pi / 2

r_0 = 30;
r_max = 50;
s_max = pi / 2;
tc = @(s) r_0 + (r_max - r_0) * cos(s - s_max);

% but now the reaching angle s will be time-dependent with the following form:
% equation 2: s(t) = sqrt(t) * pi;
% where t is the time in seconds. 0 <= t <= 1

s_t = @(t) sqrt(t) * pi;
s_inhom = zeros(1, 1000);
for i = 1:1000
    s_inhom(i) = s_t(i / 1000);
end

% (a) Generate 100 spike trains, each 1 second in duration
% , according to an inhomogeneous Poisson process with a firing rate profile defined by equation 1 and 2
% Plot 5 of the generated spike trains.


spikes_inhom = zeros(100, 1000);

% get max lambda;
max_lambda = max(tc(s_inhom));
for i = 1:100
    spikeN = poissrnd(max_lambda);
    % generate spike times
    spikeTimes = rand(1, spikeN);

    for j = 1:spikeN
        U = rand;
        if U < tc(s_inhom(ceil(spikeTimes(j) * 1000))) / max_lambda
            spikes_inhom(i, ceil(spikeTimes(j) * 1000)) = 1;
        end
    end
end

figure;
for i = 1:5
    X = find(spikes_inhom(i, :));
    plot(X, i * ones(1, length(X)), 'k.')
    hold on
end
set(gca, 'YTick', []);
ylim([0.5 5.5]);
xlabel("Time (msec)")
saveas(gcf, './results/3-a.jpg'); close all;

% (b) Plot the spike histogram by taking spike counts in non-overlapping 20 ms bins, 
% then averaging across the 100 trials. The spike histogram should have firing rate (in spikes / second)
% as the vertical axis and time (in msec, not time bin index) as the horizontal axis.
% Plot the expected firing rate profile defined by equation 1 and 2 on the same plot.
% Does the spike histogram agree with the expected firing rate profile?

spikeCounts20msInhom = zeros(100, 50);
for i = 1:100
    spikeCounts20msInhom(i, :) = sum(reshape(spikes_inhom(i, :), 20, 50), 1);
end
Y = mean(spikeCounts20msInhom, 1) * 1000 / 20;
X = (10:20:1000) / 1000;

figure;
bar(X, Y)
hold on
plot(X, tc(s_inhom(10:20:1000)), 'g', 'LineWidth', 2)
ylabel("f (Hz)")
xlabel("Time (msec)")
saveas(gcf, './results/3-b.jpg'); close all;

% (c) For each trial, count the number of spikes across the entire trial.
% Plot the normalized distribution of spike counts (i.e. normalized so that the area under the distribution equals one).
% Fit a poisson distribution to the empirical distribution and plot it on top of the empirical distribution.

spikeCountsInhom = sum(spikes_inhom, 2);
figure;
histogram(spikeCountsInhom, 'Normalization', 'probability');
hold on
x = 0:1:max(spikeCountsInhom);
y = poisspdf(x, mean(tc(s_inhom)));
plot(x, y, 'r', 'LineWidth', 2)
xlabel("Spike Counts")
ylabel("Probability")
saveas(gcf, './results/3-c.jpg'); close all;


% (d) Plot the normalized distribution of ISIs
% Fit an exponential distribution to the empirical distribution and plot it on top of the empirical distribution.


ISIsInhom = [];
for i = 1:100
    ISIsInhom = [ISIsInhom, diff(find(spikes_inhom(i, :)))];
end
ISIsInhom = ISIsInhom / 1000;
figure;
histogram(ISIsInhom, 'Normalization', 'pdf')
hold on
x = 0:0.001:max(ISIsInhom);
y = exppdf(x, 1 / mean(tc(s_inhom)));
plot(x, y, 'r', 'LineWidth', 2)
xlabel("ISIs (s)")
ylabel("Probability Density")
saveas(gcf, './results/3-d.jpg'); close all;