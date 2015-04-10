function [img,X,Xtrain,y] = loadCOLOR

%addpath ../Data/Images

img = double(imread('image.jpg'))/255.0;
m = size(img,1); n = size(img,2);

%figure(1);image(img);

forg_mask = double(imread('mask0.jpg'))/255.0;
back_mask = double(imread('mask1.jpg'))/255.0;
%figure(2);image(forg_mask.*img);
%figure(3);image(back_mask.*img);

R = reshape( img(:,:,1),[m*n 1] );
G = reshape( img(:,:,2),[m*n 1] );
B = reshape( img(:,:,3),[m*n 1] );

X = [R G B];

forg = sum( forg_mask,3 );
forg = reshape( forg , [m*n 1] );
forg_pix =  find( forg > .5 );

back = sum( back_mask,3 );
back = reshape( back , [m*n 1] );
back_pix =   back > .5 ;

Xtrain = [X(forg_pix,:);X(back_pix,:)];

y = zeros(size(Xtrain,1),1);
y(1:size(forg_pix,1),1) = 1;