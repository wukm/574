function [alpha,beta] = LOGClassify(X,y,lambda)
% classify a system (X,y) by minimizing a logarithmic energy
alpha = 0; beta = zeros(size(X,2),1);

flag = 0; dt = .00001; iter = 1; TOL = .001; itmax = 20000;
while(flag == 0),
    
    iter = iter + 1;
    
    p = GRADIENT(alpha,beta,X,y,lambda);
    
    %find dt
    dt = LINE_SEARCH(dt,p,alpha,beta,X,y,lambda);
    
    % update alpha and beta via (2)
    alpha = alpha - dt*p(1);
    beta = beta - dt*p(2:end);
    
   
    if( norm(p) < TOL || iter > itmax ),
        flag = 1;
    end;
    
end;
E = ENERGY(alpha, beta, X, y, lambda);
fprintf('LOGClassify terminated after %d iterations with final energy %f\n', iter, E) 

function p = GRADIENT(alpha,beta,X,y,lambda)

    d = size(beta,1);
    p = zeros(d+1,1);
    
    z = alpha+(X*beta);
    
    v1 = (exp(z) ./ (1+exp(z))).*(1-y);
    v2 = (exp(-z) ./ (1+exp(-z))).*y;
    v = v1 -v2;

  
    
    % p(1) should be the derivative of E with respect to alpha
    p(1) = sum(v);
    
    % p(2:end) should be the gradient of E with respect to beta
    p(2:end) = X'*v + lambda*beta;
    
function dt = LINE_SEARCH(dt,p,alpha,beta,X,y,lambda)

    dt = 2.0*dt; % make a guess for the timestep
    
    E = ENERGY(alpha,beta,X,y,lambda);
    
    % update alpha and beta via (2)
    alpha_new = alpha - dt*p(1);
    beta_new = beta - dt*p(2:end);

    E_new = ENERGY(alpha_new,beta_new,X,y,lambda);
    
    while( E_new > E - .5*dt*(p'*p) ), % then check inequality (3)
        
        % take a smaller timestep and try again
        dt = 0.5*dt;
        alpha_new = alpha - dt*p(1);
        beta_new = beta - dt*p(2:end);
        E_new = ENERGY(alpha_new,beta_new,X,y,lambda);
                              
    end;

function E = ENERGY(alpha,beta,X,y,lambda)
    % E = sum_{i=1}^{n}[ (log(1+exp(alpha + <x_i,beta>))(1-y_i)
    %                + (log(1+exp(-(alpha + <x_i,beta>)))y_i ]
    % compute the energy via (1)
    z = alpha + X*beta;
    E1 = sum( log(1+exp(z)).*(1-y));
    E2 = sum( log(1+exp(-z)).*(y));
    
    E = E1 + E2 + (lambda/2)*beta'*beta;
