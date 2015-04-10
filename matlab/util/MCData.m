function [M,B,Xex] = MCData(n,r,pct)

U = randn(n,r);
V = randn(n,r);
Xex = U*V';

q = randperm( n*n );

inds = q( 1:ceil(pct*n*n) ); %sample pct of entries
[is,js] = ind2sub(size(Xex),inds);
M = full( sparse( is , js , 1.0 , n , n ) );
B = M.*Xex;