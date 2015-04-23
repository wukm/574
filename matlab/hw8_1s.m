% semi-supervised k-means on MNIST

% load a randomly selected subset of 7s and 9s from MNIST
load('MNIST_MEDIUM.mat','img','Cact');

ind = [1:50,501:550]; % these are arbitraryish
lab = Cact(ind)

accuracies = [];
% do 200 trials of k-means and compare accuracies
for i = [1:200],
    Cest = KMEANST(img,2,ind,lab); % random initialization
    m = ACCURACY(Cest,Cact,2);
    fprintf('trial %d:\taccuracy: %f\n', i, m);
    accuracies(i) = m;
end;

fp = fopen('hw8_1s.txt', 'w');
fprintf(fp, '200 iterations of semi-supervised K-means performed\n');
fprintf(fp, 'max accuracy:\t%f%%\n', max(accuracies));
fprintf(fp, 'min accuracy:\t%f%%\n', min(accuracies));
fprintf(fp, 'mean:\t%f%%\n', mean(accuracies));
fclose(fp);