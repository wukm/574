images = loadMOTION(1:10); % this is a 10x1 cell of 512x512 matrices

% this series of arguments results in a (512*512,10) matrix where each
% column is a flattened image

img = reshape(images, 1,[]); % a 10x1 cell
 % flatten each array in the cell
img = cellfun(@(m) reshape(m,[],1), img, 'UniformOutput', false);
img = cell2mat(img);

[u,s,v] = svds(img,1); % do SVD to get largest singular value

% then this is the rank 1 approximation of the system
approx = u*s*v';

% third dimension is 
bgs = reshape(approx, 512,512,10);

% a 1x1x10 cell
bgs = num2cell(bgs, [2 1]);

% now in the same shape as images
bgs = reshape(bgs,10,1);

% this is easier
fgs = img - approx;

fgs = reshape(fgs, 512,512,10);
fgs = num2cell(fgs, [2 1]);
fgs = reshape(fgs,10,1);

