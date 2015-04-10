function viewCLASSES(img,y)

m = size(img,1); n = size(img,2);

forg = double( y > .5 );
forg = reshape(forg,[m n]);
forg = repmat(forg,[1 1 3]);

figure(1);image(forg.*img);

back = double( y <= .5 );
back = reshape(back,[m n]);
back = repmat(back,[1 1 3]);

figure(2);image(back.*img);