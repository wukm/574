function [docs,lab,vocab] = load20NEWS(groups,type)

if( strcmp(type,'test') ),
    X = dlmread('test.data');
    T = dlmread('test.label');
elseif( strcmp(type,'train') ),
    X = dlmread('train.data');
    T = dlmread('train.label');
elseif( strcmp(type,'all') ),
    X1 = dlmread('test.data');
    T1 = dlmread('test.label');
    X2 = dlmread('train.data');
    T2 = dlmread('train.label');
    X2(:,1) = X2(:,1) + size(T1,1);
    X = [X1; X2];
    T = [T1;T2];
else
    error('Please specify test,train or all');
end;

docs = X;
lab = T;

iv = docs(:,1);
jv = docs(:,2);
vv = docs(:,3);
docs = sparse(iv,jv,vv,max(iv),61188);

idx = bsxfun(@eq,lab,groups);
docs = docs( sum(idx,2) == 1 , :);
lab = lab( sum(idx,2) == 1 );

fid = fopen('_vocabulary.txt');
vocab = textscan(fid,'%s');
fclose(fid);
vocab = vocab{1};
