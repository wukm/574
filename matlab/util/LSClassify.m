function [alpha,beta] = LSClassify( Xtrain , y , lambda)
    %{
    Perform a least squares classification.

    author: Luke Wukmer
    Written for MATH 579, Spring 2015

    INPUT

    Xtrain: an mxn matrix. each of the m rows represents a data point in the
        system. this is the training data.
    y:      an mx1 vector, whose ith component is the 'class label' (0 or 1)
        of the ith data point (corresponding to the mth row of Xtrain)
    lambda: a parameter.

    OUTPUT:
    The components of the separating hyperplane f(x) = alpha + <beta, x>
    Then a data point x_ will be class 1 when f(x_) > .5, else class 0.

    alpha:  a scalar
    beta:   an nx1 vector
    %}

    % find the mean along columns of Xtrain. In other words, x_bar
    % is the componentwise average of all training points and has shape 1xn 
    xbar = mean(Xtrain, 1);

    % similarly, y_bar is the average class label. a scalar.
    ybar = mean(y);

    % this broadcasts the 1xn shape across Xtrain with the operation 'minus'
    Xtilde = bsxfun(@minus, Xtrain, xbar);

    % but broadcasting works normally like this
    ytilde = y - ybar;

    % SVD decompose the mean centered system
    [u, s, v] = svd(Xtilde, 'econ');

    % c values
    s_values = diag(s,0);
    c = s_values ./ (s_values.*s_values + lambda);

    % alpha and beta
    beta = v * (c .* (u'*ytilde));
    alpha = ybar - xbar*beta;