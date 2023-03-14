function plot_ellipse(mu, sigma, color)
    [V,D] = eig(sigma);
    theta = 0:0.01:2*pi;
    x = sqrt(D(1,1))*cos(theta);
    y = sqrt(D(2,2))*sin(theta);
    z = V*[x;y];
    plot(mu(1)+z(1,:), mu(2)+z(2,:), color);
end