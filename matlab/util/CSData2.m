function [A,b,x_ex] = CSData2(m,n,s)

load('A.mat','A');
n = 500;

rows = randperm(n); rows = rows(1:m); 

A = A(rows,:); % sensing matrix

x_ex = zeros(n,1);
inds = randperm(n);
x_ex(inds(1:s)) = 1.0*randn(s,1); % exact signal
b = A*x_ex; % measurments