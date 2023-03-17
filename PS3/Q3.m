clear all; close all;
load("ps3_simdata.mat");
% The file has a single variable named
% trial, which is a structure of dimensions (20 data points) × (3 classes). The nth data
% point for the kth class is a two-dimensional vector trial(n,k).x, where n = 1, . . . , 20
% and k = 1, 2, 3.

nneurons = length(trial(1,1).x);
[ntrials, nclasses] = size(trial);
% nneurons = 2, ntrials = 20, nclasses = 3

% a. Plot the data points in a two-dimensional space. For classes k = 1, 2, 3,
% use a red ×, green +, and blue ◦ for each data point, respectively. Then, set the
% axis limits of the plot to be between 0 and 20. (Matlab tip: call axis([0 20 0
% 20]))

figure;
hold on;
for k = 1:nclasses
    for n = 1:ntrials
        if k == 1
            plot(trial(n,k).x(1), trial(n,k).x(2), 'rx');
        elseif k == 2
            plot(trial(n,k).x(1), trial(n,k).x(2), 'g+');
        else
            plot(trial(n,k).x(1), trial(n,k).x(2), 'bo');
        end
    end
end
axis([0 20 0 20]);
hold off;
saveas(gcf, 'ps3_1a.png'); close all;

% b. Find the ML model parameters in 3 different model
% 1) gaussian distribution with shared covariance matrix
% 2) gaussian distribution with individual covariance matrix
% 3) poisson distribution with individual lambda in each class and neuron

parameters = cell(3,1);
for m = 1:3
    if m == 1
        % shared covariance matrix
        mu = zeros(nneurons, nclasses);
        sigma = zeros(nneurons, nneurons);
        for k = 1:nclasses
            mu(:,k) = mean([trial(:,k).x], 2);
            sigma = sigma + cov([trial(:,k).x]');
        end
        sigma = sigma / nclasses;
        parameters{m} = {mu, sigma};
    elseif m == 2
        % individual covariance matrix
        mu = zeros(nneurons, nclasses);
        sigma = cell(nclasses,1);
        for k = 1:nclasses
            mu(:,k) = mean([trial(:,k).x], 2);
            sigma{k} = cov([trial(:,k).x]');
        end
        parameters{m} = {mu, sigma};
    else
        % poisson distribution
        lambda = zeros(nneurons, nclasses);
        for k = 1:nclasses
            lambda(:,k) = mean([trial(:,k).x], 2);
        end
        parameters{m} = {lambda};
    end
end

% c.  For each class, plot the ML mean using a solid dot of the appropriate
% color.

for m=1:3
    figure;
    hold on;
    for k = 1:nclasses
        for n = 1:ntrials
            if k == 1
                plot(trial(n,k).x(1), trial(n,k).x(2), 'rx');
            elseif k == 2
                plot(trial(n,k).x(1), trial(n,k).x(2), 'g+');
            else
                plot(trial(n,k).x(1), trial(n,k).x(2), 'bo');
            end
        end
    end

    for k = 1:nclasses
        if k == 1
            plot(parameters{m}{1}(1,k), parameters{m}{1}(2,k), 'r.', 'MarkerSize', 30);
        elseif k == 2
            plot(parameters{m}{1}(1,k), parameters{m}{1}(2,k), 'g.', 'MarkerSize', 30);
        else
            plot(parameters{m}{1}(1,k), parameters{m}{1}(2,k), 'b.', 'MarkerSize', 30);
        end
    end
    axis([0 20 0 20]);
    hold off;
    saveas(gcf, sprintf('ps3_1c_%d.png', m)); close all;
end

% d. For each class, plot the ML covariance using an ellipse of the appro
% priate color.

for m=1:2
    figure;
    hold on;
    for k = 1:nclasses
        for n = 1:ntrials
            if k == 1
                plot(trial(n,k).x(1), trial(n,k).x(2), 'rx');
            elseif k == 2
                plot(trial(n,k).x(1), trial(n,k).x(2), 'g+');
            else
                plot(trial(n,k).x(1), trial(n,k).x(2), 'bo');
            end
        end
    end

    if m == 1
        % shared covariance matrix
        for k = 1:nclasses
            if k == 1
                plot_ellipse(parameters{m}{1}(:,k), parameters{m}{2}, 'r');
            elseif k == 2
                plot_ellipse(parameters{m}{1}(:,k), parameters{m}{2}, 'g');
            else
                plot_ellipse(parameters{m}{1}(:,k), parameters{m}{2}, 'b');
            end
        end
    else
        % individual covariance matrix
        for k = 1:nclasses
            if k == 1
                plot_ellipse(parameters{m}{1}(:,k), parameters{m}{2}{k}, 'r');
            elseif k == 2
                plot_ellipse(parameters{m}{1}(:,k), parameters{m}{2}{k}, 'g');
            else
                plot_ellipse(parameters{m}{1}(:,k), parameters{m}{2}{k}, 'b');
            end
        end
    end
    axis([0 20 0 20]);
    hold off;
    saveas(gcf, sprintf('ps3_1d_%d.png', m)); close all;
end

% e. Plot multi-class decision boundaries corresponding to the decision rule
% k = argmaxk p(x|k) for each model. For each model, use a different color
% and label each decision region with the appropriate class k.

for m=1:3
    figure;
    hold on;
    
    if m == 1
        % shared covariance matrix
        % define the probability at x for each class
        p_x_1 = @(x) parameters{m}{1}(:,1)' * inv(parameters{m}{2}) * x - 0.5 * parameters{m}{1}(:,1)' * inv(parameters{m}{2}) * parameters{m}{1}(:,1);
        p_x_2 = @(x) parameters{m}{1}(:,2)' * inv(parameters{m}{2}) * x - 0.5 * parameters{m}{1}(:,2)' * inv(parameters{m}{2}) * parameters{m}{1}(:,2);
        p_x_3 = @(x) parameters{m}{1}(:,3)' * inv(parameters{m}{2}) * x - 0.5 * parameters{m}{1}(:,3)' * inv(parameters{m}{2}) * parameters{m}{1}(:,3);
    elseif m == 2
        % individual covariance matrix
        % define the probability at x for each class
        p_x_1 = @(x) -0.5 * log(det(parameters{m}{2}{1})) - 0.5 * (x - parameters{m}{1}(:,1))' * inv(parameters{m}{2}{1}) * (x - parameters{m}{1}(:,1));
        p_x_2 = @(x) -0.5 * log(det(parameters{m}{2}{2})) - 0.5 * (x - parameters{m}{1}(:,2))' * inv(parameters{m}{2}{2}) * (x - parameters{m}{1}(:,2));
        p_x_3 = @(x) -0.5 * log(det(parameters{m}{2}{3})) - 0.5 * (x - parameters{m}{1}(:,3))' * inv(parameters{m}{2}{3}) * (x - parameters{m}{1}(:,3));
    else
        % poisson distribution
        % define the probability at x for each class
        p_x_1 = @(x) sum(log(parameters{m}{1}(:,1)) .* x) - sum(parameters{m}{1}(:,1));
        p_x_2 = @(x) sum(log(parameters{m}{1}(:,2)) .* x) - sum(parameters{m}{1}(:,2));
        p_x_3 = @(x) sum(log(parameters{m}{1}(:,3)) .* x) - sum(parameters{m}{1}(:,3));
    end
    % find the decision boundary
    [x1, x2] = meshgrid(0:0.1:20, 0:0.1:20);
    x = [x1(:)'; x2(:)'];
    p_x = zeros(3, size(x, 2));
    for i = 1:size(x, 2)
        p_x(1, i) = p_x_1(x(:, i));
        p_x(2, i) = p_x_2(x(:, i));
        p_x(3, i) = p_x_3(x(:, i));
    end
    [~, k] = max(p_x);
    k = reshape(k, size(x1));
    X1 = cell(1, 3);
    X2 = cell(1, 3);
    for i = 1:size(k, 1)
        for j = 1:size(k, 2)
            if k(i, j) == 1
                X1{1} = [X1{1} x1(i, j)];
                X2{1} = [X2{1} x2(i, j)];
            elseif k(i, j) == 2
                X1{2} = [X1{2} x1(i, j)];
                X2{2} = [X2{2} x2(i, j)];
            else
                X1{3} = [X1{3} x1(i, j)];
                X2{3} = [X2{3} x2(i, j)];
            end
        end
    end
    
    for i = 1:3
        % plot with the opacity = 0.5
        if i == 1
            scatter(X1{i}, X2{i}, 10, 'r', 'filled', 'MarkerFaceAlpha', 0.2);
        elseif i == 2
            scatter(X1{i}, X2{i}, 10, 'g', 'filled', 'MarkerFaceAlpha', 0.2);
        else
            scatter(X1{i}, X2{i}, 10, 'b', 'filled', 'MarkerFaceAlpha', 0.2);
        end
    end

    for k = 1:nclasses
        for n = 1:ntrials
            if k == 1
                plot(trial(n,k).x(1), trial(n,k).x(2), 'rx');
            elseif k == 2
                plot(trial(n,k).x(1), trial(n,k).x(2), 'g+');
            else
                plot(trial(n,k).x(1), trial(n,k).x(2), 'bo');
            end
        end
    end

    if m == 1
        % shared covariance matrix
        for k = 1:nclasses
            if k == 1
                plot_ellipse(parameters{m}{1}(:,k), parameters{m}{2}, 'r');
            elseif k == 2
                plot_ellipse(parameters{m}{1}(:,k), parameters{m}{2}, 'g');
            else
                plot_ellipse(parameters{m}{1}(:,k), parameters{m}{2}, 'b');
            end
        end
    else
        % individual covariance matrix
        for k = 1:nclasses
            if k == 1
                plot_ellipse(parameters{m}{1}(:,k), parameters{m}{2}{k}, 'r');
            elseif k == 2
                plot_ellipse(parameters{m}{1}(:,k), parameters{m}{2}{k}, 'g');
            else
                plot_ellipse(parameters{m}{1}(:,k), parameters{m}{2}{k}, 'b');
            end
        end
    end
    
    axis([0 20 0 20]);

    hold off;
    saveas(gcf, sprintf('ps3_1e_%d.png', m)); close all;
end