function W = gaussian_kernel(X,sigma)
    % make the gaussian kernel with parameter sigma
    %{
    INPUT
        X:
            an nxd matrix (typically the 'system')
        sigma:
            a parameter
    OUTPUT
        W:
            an nxn gaussian kernel (defined below)

    This creates the so-called "gaussian kernel," a similarity matrix with
    components
        W_ij := exp{ -(||x_i - x_j||_2)^2 / (2*sigma^2) }
    
    Note the datapoints x_i, x_j are assumed to be *rows* of X.
    
    Specifically, W is constructed via the relation
        W = (Q + XX') + (Q + XX')'
        
    where Q is the nxn matrix s.t. Q_ij = (||x_i||_2)^2
    
    %}
    
    % construct in multiple steps. matlab can freeze for large n
    W = X.*X;
    W = repmat(sum(W,2), 1, size(X,1));
    W = W - X*X';
    W = W + W';
    W = -W / (2*sigma^2);
    W = exp(W);