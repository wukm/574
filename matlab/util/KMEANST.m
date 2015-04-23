% transductive k-means (semi-supervised)
% it's probably possible to overload the original KMEANS function so these
% can be combined. lol matlab

function Cest = KMEANST(X,R,ind,lab)
    % Lloyd's algorithm
    % k-means with random initial guesses, one time through
    n = size(X,1);
    Cest = randi(R,n,1); % make a random initial partition
    Cest(ind) = lab; % and set any labels we know
    flag = 0;
    
    while ( flag == 0 ),
        F = full( sparse(1:n,Cest,1,n,R) );
        F = bsxfun(@rdivide, F, sum(F,1));
        M = X'*F; % compute representatives
        D = 2*X*M - ones(n,1)*sum(M.*M,1); % compute distance to each representative
        Cold = Cest;
        [~,Cest] = max(D,[],2); % assign points to classes based on these distances
        Cest(ind) = lab; % and set any labels we know
        if ( max(max( abs(Cest - Cold))) < 1e-7 )
            flag = 1;
        end;
    end;