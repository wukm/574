load('MNIST_MEDIUM.mat','img','Cact');

sigma = 1.25;

% make W
Q = repmat(sum(img'*img,2),size(img,2));
W = Q - X*X';
W = W + W';
W = -M / (2*sigma^2);
W = exp(W);