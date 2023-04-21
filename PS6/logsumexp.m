function result = logsumexp(input, dim)
    max_input = max(input, [], dim);
    result = max_input + log(sum(exp(bsxfun(@minus, input, max_input)), dim));
end