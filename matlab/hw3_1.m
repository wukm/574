[img, X, Xtrain, y] = loadCOLOR;
lambda = .05; % change this to three values: .005, .05, .5
[alpha, beta] = LOGClassify(Xtrain,y,.05);

y_est = double( (alpha + X*beta) > 0 );

viewCLASSES(img, y_est + .5);