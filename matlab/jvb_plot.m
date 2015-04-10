function jvb_plot
% returns a non linearly-separable system that JVB loves so much

% you could specify any of these but eh
n = 400; % how many points in each class you want
%there should be a different way to do this i think
R1 = .3;
R2 = 10.;

r1 = randn(n,1); r2 = randn(n,1)+R2;
th1 = 2*pi*rand(n,1); th2 = 2*pi*rand(n,1);

X1 = R1*[sqrt(r1).*cos(th1) sqrt(r1).*sin(th1)];
X2 = R1*[sqrt(r2).*cos(th2) sqrt(r2).*sin(th2)];

X = cat(1, X1, X2);
%scatter(X(:,1), X(:,2));
