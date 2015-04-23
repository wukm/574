function W = gaussian_kernel(X,sigma)
    % make the gaussian kernel with parameter sigma
    % math docstrings to follow, sorry
    % here the datapoints x_i, x_j are assumed to be *rows*
    
    % doing this in a bunch of steps because of memory.
    W = X.*X;
    W = repmat(sum(W,2), 1, size(X,1));
    W = W - X*X';
    W = W + W';
    W = -W / (2*sigma^2);
    W = exp(W);