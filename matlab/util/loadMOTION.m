function m = loadMOTION(images, show)

n = size(images,2);
m = cell(n,1);

if( max(images) > 10 || min(images) < 1 )
    error('Frame indices run from 1 to 10');
end;

for i=1:n,
    loadstr = strcat('motion_',num2str(images(i)),'.tiff');
    m{i} = double(imread(loadstr))/255.0;
    %figure(i);imagesc(m{i});colormap gray;
end;