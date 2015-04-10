[docs, lab, vocab] = load20NEWS([10,11], 'train');
[docs_t, lab_t] = load20NEWS([10,11], 'test');

y = double(lab == 10); 
binary = double(lab_t == 10);

% uncomment this to do feature extraction
%{
docs = double(docs > 0);
docs_t = double(docs_t > 0);
%}

lambda = 10.; % want to test for lambda in [1. 10. 100.]
for lambda = [1.,10.,100.]
    fprintf('classifying with lambda=%f\n', lambda);
    [alpha, beta] = LOGClassify(docs, y, lambda);
    binary_est = double( alpha + docs_t*beta > 0 );
    m = ACCURACY(binary_est, binary, 2);
    fprintf('accuracy: %f\n', m);
    [~, indices] = sort(beta, 'descend');
    
    class0 = vocab(indices(1:10));
    class1 = vocab(indices(end-10:end));
    fprintf('class 0 words: %s\n', strjoin(class0, ' '));
    fprintf('class 1 words: %s\n', strjoin(class1, ' '));
    fprintf('---------------\n')
end;
    
