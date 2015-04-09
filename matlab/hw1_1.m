%{
write description of problem here.

when this is done, you can view any of the results individually e.g.
imagesc(fgs{4}); colormap gray;
or save with something like this (didn't test, don't know how to change
path for saving or do string manipulation yet

for i=1:numel(images)
    imwrite(fgs{i}, whatever_the_filename_is.tiff);
    imwrite(bgs{i}, whatever_the_filename_is.tiff);
end;

and do some string manipulation to get the right filename, whatever that is

btw, apparently if the matrix in imwrite is doubles in [0,1] and you save
in tiff (and some other formats), it will automatically convert it to
grayscale scaled 0,255. neato.
%}
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



bgs = columns_to_image_list(bgs,512,512);

% subtracting the rank1 background from each image yields a foreground 
fgs = img - approx;

% do the same stuff on columns of the foreground matrix
fgs = reshape(fgs, 512,512,10);
fgs = num2cell(fgs, [2 1]);
fgs = reshape(fgs,10,1);

fgs = cellfun(@transpose, fgs, 'UniformOutput', false);