function logpdf = logmvnpdf(x, mu, Sigma)
    d = length(mu);
    x_minus_mu = x - mu;
    invSigma = inv(Sigma);
    logpdf = -0.5 * (d * log(2 * pi) + log(det(Sigma)) + sum((x_minus_mu * invSigma) .* x_minus_mu, 2));
end
