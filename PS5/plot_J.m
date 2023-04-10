function plot_J(J_E_values, J_M_values, name)
    % Input:
    % J_E_values: a max_iter x 1 vector containing the objective function values for all E-steps
    % J_M_values: a max_iter x 1 vector containing the objective function values for all M-steps
    % max_iter: maximum number of iterations

    % Create x-axis values for the E-step and M-step
    x_E = 0.5:1:(length(J_E_values) - 0.5);
    x_M = 1:1:length(J_M_values);


    figure;
    hold on;
    % Plot J after the E-step (blue, empty circles)
    plot(x_E, J_E_values, 'o', 'MarkerEdgeColor', 'blue', 'MarkerFaceColor', 'none', 'DisplayName', 'J after E-step');

    % Plot J after the M-step (red, empty circles)
    plot(x_M, J_M_values, 'o', 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'none', 'DisplayName', 'J after M-step');

    % Connect each point to the closest marker (depending on x-axis) with a black line
    for i = 1:length(J_E_values)
        x = [x_E(i), x_M(i)];
        y = [J_E_values(i), J_M_values(i)];
        plot(x, y, '-k');
    end

    for i = 1:length(J_M_values)-1
        x = [x_M(i), x_E(i+1)];
        y = [J_M_values(i), J_E_values(i+1)];
        plot(x, y, '-k');
    end

    % Add labels and legend
    xlabel('Iteration');
    ylabel('Objective Function J');
    saveas(gcf, name); close all;

    disp(J_M_values(end));
end