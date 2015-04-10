function [voice,lab] = loadISOLET(letters,type)

if( strcmp(type,'test') ),
    X = dlmread('isolet.test');
    T = X(:,end);
    X(:,end) = [];
elseif( strcmp(type,'train') ),
    X = dlmread('isolet.training');
    T = X(:,end);
    X(:,end) = [];
elseif( strcmp(type,'all') ),
    X1 = dlmread('isolet.training');
    T1 = X1(:,end);
    X1(:,end) = [];
    X2 = dlmread('isolet.test');
    T2 = X2(:,end);
    X2(:,end) = [];
    X = [X1;X2];
    T = [T1;T2];
else
    error('Please specify test,train or all');
end;

voice = X;
lab = T;

idx = bsxfun(@eq,lab,letters);
voice = voice( sum(idx,2) == 1 , :);
lab = lab( sum(idx,2) == 1 );