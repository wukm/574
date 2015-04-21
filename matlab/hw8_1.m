% k means on MNIST

% load a randomly selected subset of 7s and 9s from MNIST
load('MNIST_MEDIUM.mat','img','Cact');

accuracies = [];
% do 200 trials of k-means and compare accuracies
for i = [1:200],
    Cest = KMEANS(img,2); % random initialization
    m = ACCURACY(Cest,Cact,2);
    fprintf('trial %d:\taccuracy: %f\n', i, m);
    accuracies(i) = m;
end;

fprintf('200 iterations of K-means performed\n');
fprintf('max accuracy:\t%f%%\n', max(accuracies));
fprintf('min accuracy:\t%f%%\n', min(accuracies));
fprintf('mean:\t%f%%\n', mean(accuracies));
fprintf('variance:\t%f\n', var(accuracies)); 