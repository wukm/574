d = 2;

% images is a 7844x784 matrix -- each row is a vectorized image of d
[images, labels] = loadMNIST(d,'all');

[u,s,v] = svds(images, 2);

leads = columns_to_image_list(v,28,28);

figure(1); imagesc(leads{1}); colormap gray;
figure(2); imagesc(leads{2}); colormap jet;

% index of largest/smallest left singular vector u
% check if these are correct. does it max over... norm?
[~, imax] = max(u(:,2));
[~, imin] = min(u(:,2));

maxu = reshape(images(imax,:), [28 28]);
minu = reshape(images(imin,:), [28 28]);

figure(3); imagesc(maxu); colormap gray;
figure(4); imagesc(minu); colormap gray;