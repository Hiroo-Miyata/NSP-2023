function plot_cluster(x, label, k, mu2, Vthresh, name)

    figure;
    plot(1:31, x(:,label==k), "Color", "k"); % plot snippets
    hold on;
    plot(mu2(:,k), 'r', 'LineWidth', 1.5); % plot cluster center
    plot([1 31], [Vthresh Vthresh], 'b', 'LineWidth', 1.5); % plot threshold
    xticks([1 10 19 31]);
    xticklabels([-0.3 0 0.3 0.7]);
    xlim([1 31]);
    xlabel('Time (ms)');
    ylabel('Voltage (µV)');
    ylim([-400 1000]);
    title("Voltage vs. Time Plot (Cluster" + k + ")");
    saveas(gcf, name);

end