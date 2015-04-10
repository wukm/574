function [alpha,beta] = LOGClassify(X,y,lambda)

alpha = 0; beta = zeros(size(X,2),1);

flag = 0; dt = .001; iter = 1; TOL = .001; itmax = 20000;
while(flag == 0),
    
    iter = iter + 1;
    
    p = GRADIENT(alpha,beta,X,y,lambda);
    
    %find dt
    dt = LINE_SEARCH(dt,p,alpha,beta,X,y,lambda);
    
    % update alpha and beta via (2)
    %    
    %
    
   
    if( norm(p) < TOL || iter > itmax ),
        flag = 1;
    end;
    
end;

function p = GRADIENT(alpha,beta,X,y,lambda)

    d = size(beta,1);
    p = zeros(d+1,1);
    
    % p(1) should be the derivative of E with respect to alpha
    p(1) =
    
    % p(2:end) should be the gradient of E with respect to beta
    p(2:end) =
    
function dt = LINE_SEARCH(dt,p,alpha,beta,X,y,lambda)

    dt = 2.0*dt; % make a guess for the timestep
    
    E = ENERGY(alpha,beta,X,y,lambda);
    
    % update alpha and beta via (2)
    %    
    %

    E_new = ENERGY(alpha_new,beta_new,X,y,lambda);
    
    while( E_new > E - .5*dt*(p'*p) ), % then check inequality (3)
        
        % take a smaller timestep and try again
        %
        %
        %                      
        
        E_new = ENERGY(alpha_new,beta_new,X,y,lambda);
                              
    end;

function E = ENERGY(alpha,beta,X,y,lambda)
    
    % compute the energy via (1)
    E = 
