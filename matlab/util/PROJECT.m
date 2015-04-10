function x = PROJECT(x,y,C)

% compute the projection of the n-vector x onto the convex
% set K defined via
%     C >= x_i y_i >= 0,
%     sum(x) = 0
% using Dykstra's alternating projection algorithm

p = zeros(size(x,1),1);
q = zeros(size(x,1),1);

flag = 0;iter = 1;

while(flag == 0),
    
    p_old = p; q_old = q;
    
    %project and calculate increment
    z = PROJ_C(x+p); 
    p = x + p - z;
    
    %project and calculate increment
    x = PROJ_D(z+q,y,C);
    q = z + q - x;
    
    nrm = ( (p-p_old)' )*(p - p_old) + ( (q-q_old)' )*(q - q_old);
    
    iter = iter + 1;
    
    if( nrm < 1e-15 )
        flag = 1;
    end;
    
end;

function m = PROJ_C(x)
    m = x - mean(x);
    
function m = PROJ_D(x,y,C)

    if( strcmp(C,'inf') ),
        m = x.*double( (x.*y) >= 0 );
    elseif( C > 0 ),
        cond1 = double( x.*y >= 0 );
        cond2 = double( x.*y <= C );
        m = x.*cond1.*cond2 + C*y.*(1-cond2);
    else
        error('Invalid Penalty Parameter');
    end;