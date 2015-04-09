% LSClassify on MNIST

% do this for the two sets
% digits = [1,6]
digits = [4,9]

[img, lab] = loadMNIST(digits, 'train');
y = double(lab == digits(2));
lambda = 20.0;
[alpha, beta] = LSClassify(img, y, lambda);

[img_t, lab_t] = loadMNIST(digits, 'test');

binary = double(lab_t == digits(2)); % the actual classes for the testing set
binary_est = double(alpha + img_t*beta > .5); % the classes computed

% got 99.52% for 1 & 6
% and 96.63% for 4 & 9
pct = ACCURACY(binary_est + 1, binary + 1, 2);
