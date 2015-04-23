load('MNIST_MEDIUM.mat','img','Cact');
sigma = 1.25;

W = gaussian_kernel(img,sigma);
L = diag(sum(W,2),0) - W;
[V,E] = eigs(L,2,'sm');
f = V(:,1);

Cest = double(f>0) + 1;
m = ACCURACY(Cest,Cact,2);


% now supervised part
ind = [1:20,501:520]; % these are arbitrary
lab = 2.0*(Cact(ind) - 1.0) - 1.0; %

lambda = 1000.0;

b = zeros(size(W,1),1);
b(ind) = lab; % enforce known labels
M = diag( abs(b) , 0);

% (L+lambda*M)f = lambda*M*b
f = (L+lambda*M)\(lambda*M*b);

% assignment rule
Cest_s = double(f>0)+1;

ms = ACCURACY(Cest_s, Cact, 2);

fp = fopen('hw8_2.txt','w');
fprintf(fp,'unsupervised classification via spectral clustering. accuracy: %f\n', m);
fprintf(fp,'semi-supervised classification accuracy: %f\n', ms);
fclose(fp);