function [faces,labels] = loadFACES(classes,type)

X = zeros(2414,192*168);
lab = zeros(2414,1);
tot = 1;
for i=1:39,
    
    if( i == 14 )
        continue;
    end;
    
    if( i < 10 )
        loadstr = strcat('YaleB/yaleB0',num2str(i),'/');
        str = strcat('YaleB/yaleB0',num2str(i),'/*.pgm');
    else
        loadstr = strcat('YaleB/yaleB',num2str(i),'/');
        str = strcat('YaleB/yaleB',num2str(i),'/*.pgm');
    end;
    
    list = dir(str);
    if( length(list) < 1 ),
        error('Cannot Locate YaleB Data');
    end;
    
    
    for j = 1:length(list),
        
        A = double(imread(  strcat(loadstr,list(j).name)  ));
        [m n] = size(A);
        X(tot,:) = reshape(A,[1 m*n]);
        if( i < 14 )
            lab(tot) = i;
        else
            lab(tot) = i-1;
        end;
        
        tot = tot + 1;
        
    end;
    
end;

train_indices = dlmread('YaleB/train_indices.txt'); % a random subset of the full data set
Xtrain = X(train_indices,:); 
lab_train = lab(train_indices,:);

Xtest = X; Xtest(train_indices,:) = [];
lab_test = lab; lab_test(train_indices,:) = [];

if( strcmp(type,'test') )
    faces = Xtest; labels = lab_test;
elseif( strcmp(type,'train') )
    faces = Xtrain; labels = lab_train;
elseif( strcmp(type,'all') )
    faces = X; labels = lab;
else
    error('Please input test,train or all');
end;

faces = double(faces)/255.0;

idx = bsxfun(@eq,labels,classes);
faces = faces( sum(idx,2) == 1 , :);
labels = labels( sum(idx,2) == 1 );