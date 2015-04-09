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

%{
if m != d1*d2
    throw an error
end
%}

A = reshape(A, d1,d2,n);
A = num2cell(A, [2 1]);
A = reshape(A,n,1);

C = cellfun(@transpose, A, 'UniformOutput', false);