function [cluster_assignments, updated_centers, J_E_values, J_M_values] = kmeans_clustering(data, initial_centers, max_iter)

% Input:
% data: a 31 x N matrix, where N is the number of data points
% initial_centers: a 31 x K matrix, where K is the number of clusters
% max_iter: maximum number of iterations

% Output:
% cluster_assignments: an N x 1 vector containing the cluster assignments
% updated_centers: a 31 x K matrix containing the updated cluster centers
% J_E_values: a max_iter x 1 vector containing the objective function values for all E-steps
% J_M_values: a max_iter x 1 vector containing the objective function values for all M-steps

% Set the number of clusters
k = size(initial_centers, 2);

% Set the number of data points
N = size(data, 2);

% Initialize cluster_assignments, updated_centers, and objective function value arrays
cluster_assignments = zeros(N, 1);
updated_centers = initial_centers;
J_E_values = zeros(max_iter, 1);
J_M_values = zeros(max_iter, 1);

% Iterate up to max_iter times
for iter = 1:max_iter
    
    % E-Step: Assign each data point to the nearest center
    for i = 1:N
        min_distance = inf;
        for j = 1:k
            distance = norm(data(:, i) - updated_centers(:, j))^2;
            if distance < min_distance
                min_distance = distance;
                cluster_assignments(i) = j;
            end
        end
    end
    
    % Calculate the objective function J for the E-step
    J_E = 0;
    for i = 1:N
        J_E = J_E + norm(data(:, i) - updated_centers(:, cluster_assignments(i)))^2;
    end
    J_E_values(iter) = J_E;
    
    % M-Step: Update the cluster centers
    prev_centers = updated_centers;
    for j = 1:k
        cluster_points = data(:, cluster_assignments == j);
        if ~isempty(cluster_points)
            updated_centers(:, j) = mean(cluster_points, 2);
        end
    end
    
    % Calculate the objective function J for the M-step
    J_M = 0;
    for i = 1:N
        J_M = J_M + norm(data(:, i) - updated_centers(:, cluster_assignments(i)))^2;
    end
    J_M_values(iter) = J_M;
    
end

end
