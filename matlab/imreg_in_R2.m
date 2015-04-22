% toy problem of image registration (or something) in R^2

% separate this into two things ...
% you can either extract a subset closest together (good for testing)
% or get *any* random subset. just throw a flag and do subfunctions ya

clustered = 0;
N = 100; % total number of points in set
n_extracted = 5; % the number of points to extract 

scale_by = 10*rand(); % get a random scaling value

i = 1; % this is just for indexing figures, ignore

% N random pairs of doubles between 0 and 100
A = rand(2,N)*100;
% scatter(A(:,1),A(:,2),'filled')

if clustered,
    % arbitrarily choose one pair
    xo = A(:,randi(size(A,1)));

    % sort A by euclidean distance to this point
    ed = bsxfun(@minus, A, xo);
    ed = ed.*ed;
    ed = sum(ed,1);
    [~, ind] = sort(ed);

    dA = A(:,ind); % this is A with rows sorted by distance from xo
else,
    % just randomly permute the columns
    dA = A(:,randperm(size(A,2)));
end;
% this defines how to color the points.
% make the extracted ones a different color
c = zeros(size(dA,2),3);
c(1:n_extracted,3) = 1; c(1:n_extracted,2) = .5;

% plot the entire set with the 'extracted points' as a different color
figure(i); scatter(dA(1,:),dA(2,:), [], c, 'Marker', '*'); i=i+1;

% okay, so we extract these points. This is now "X" exactly, or what we
% will hope to recover again.
X_ex = dA(:,1:n_extracted);
%figure(i); scatter(X_ex(:,1),X_ex(:,2),'Marker', '*'); i=i+1;
    
% now rotate these by some random amount
theta = 2*pi*rand();
R = [cos(theta) -sin(theta) ; sin(theta) cos(theta)];

% rotate each row according to R
X = R*X_ex;
% now translate it down to the origin and scale
X = scale_by*bsxfun(@minus, X, min(X,[],2));
figure(i); scatter(X(1,:),X(2,:),[],[0,.5,1],'Marker', '*'); i=i+1;

% now solve this problem to get the inverse transformation
% T = transformation_search(A,x);

fprintf('The correct transformation is\n');
fprintf('\ttheta=%f\n',theta);
fprintf('\tk=%f\n',scale_by);
fprintf('\tv=(%f,%f)\n',(dA(:,1)-X(:,1)));





