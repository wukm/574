function [images,labels] = loadMNIST(digits,type)

if( strcmp(type,'test') ),
    X = loadImages('testing_images');
    T = loadLabels('testing_labels');
elseif( strcmp(type,'train') ),
    X = loadImages('training_images');
    T = loadLabels('training_labels');
elseif( strcmp(type,'all') ),
    X1 = loadImages('training_images');
    T1 = loadLabels('training_labels');
    X2 = loadImages('testing_images');
    T2 = loadLabels('testing_labels');
    X = [X1, X2];
    T = [T1;T2];
else
    error('Please specify test,train or all');
end;

images = X';
labels = T;

dig = bsxfun(@eq,labels,digits);
images = images( sum(dig,2) == 1 , :);
labels = labels( sum(dig,2) == 1 );

function images = loadImages(filename)
    
    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open ', filename, '']);

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, ['Bad magic number in ', filename, '']);

    numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
    numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
    numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

    images = fread(fp, inf, 'unsigned char');
    images = reshape(images, numCols, numRows, numImages);
    images = permute(images,[2 1 3]);

    fclose(fp);

    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
    images = double(images) / 255;

function labels = loadLabels(filename)

    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open ', filename, '']);

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, ['Bad magic number in ', filename, '']);

    numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

    labels = fread(fp, inf, 'unsigned char');

    assert(size(labels,1) == numLabels, 'Mismatch in label count');

    fclose(fp);
    