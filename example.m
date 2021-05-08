load('fisheriris');
X = randn(150,10);
X(:,[1 3 5 7 ])= meas;
y = species;
opt = statset('display','iter');
% Generating a stratified partition is usually preferred to
% evaluate classification algorithms.
cvp = cvpartition(y,'k',10); 
[fs,history] = sequentialfs(@classf,X,y,'cv',cvp,'options',opt);

function err = classf(xtrain,ytrain,xtest,ytest)
    yfit = classify(xtest,xtrain,ytrain,'quadratic');
    err = sum(~strcmp(ytest,yfit));
end