function C = columns_to_image_list(A, d1,d2)
% turns columns of a matrix into a cell of image matrices

%{
takes some matrix mxn A and attemps to turn each row into a d1xd2
image. (note it must be that d1*d2 =m) these images (really
still arrays) are then returned as a cell array of size nx1.

things i could implement:
    -   guess size if d2 is not specified
    -   if d1,d2 not implemented
    -   be able to pack as [d1 d2] (then just check if the first arg is
        an int or a tuple)
    - nothing; this function is great; thank you me
%}

[m,n] = size(A);

% check that we can actually form matrices with the requested dimensions
if m ~= d1*d2
    error('cannot make matrices with those dimensions for the columns provided');
end;

% okay to be honest i just mashed stuff into these functions until i got
% what i wanted. goal is to turn each column of (approx) and (img-approx)
% into a 512x512 matrix that can be viewed as an image. what follows works
% but i doubt it is the most straightforward, i just heard for loops are
% bad so here's the complicated stuff instead, enjoy

% % third dimension is each individual image (but transposed?) 
% bgs = reshape(approx, 512,512,10);
% 
% % a 1x1x10 cell
% bgs = num2cell(bgs, [2 1]);
% % this 
% bgs = cellfun(@transpose, bgs, 'UniformOutput', false);
% % now in the same shape as images
% bgs = reshape(bgs,10,1);



%{
if m != d1*d2
    throw an error
end
%}

% so now each A(:,:,n) is an individual image (but transposed)
A = reshape(A, d1,d2,n);
% this will create a 1x1xn cell of d1xd2 matrices
A = num2cell(A, [2 1]);
% make it a nx1 cell instead
A = reshape(A,n,1);

% now transpose the matrices within the cell
% also wow this function is even better than bsxfun /s
C = cellfun(@transpose, A, 'UniformOutput', false);