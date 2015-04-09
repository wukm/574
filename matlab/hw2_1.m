% solution to HW 2 question 1

% question asks for lambda = [.005,.5,10,250,800]
lambda = 800
[img,X,Xtrain,y] = loadCOLOR;
[alpha, beta] = LSClassify(Xtrain, y, .001);

y = (X * beta) + alpha;
y = double(y > .5);

viewCLASSES(img, y);
