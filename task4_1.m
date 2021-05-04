% MLPs to estimate arousal and valence.

% The MLPs take as input a set 
% of features that are selected from those described in Section 2.1 and return 
% the corresponding valence and arousal levels, respectively.

% The set of extracted features should be reduced by selecting 
% the most significant features to predict the output. One way 
% is to use the sequential feature selection (implemented by the sequentialfs
% MATLAB function), with a neural network as a criterion function 
% that assesses the accuracy of each subset of features in estimating 
% the valence (or arousal) level. The suggested maximum number of 
% features to select is 10 for each network. Once the search for the 
% best set of features is completed, the next step is to find the 
% best architecture for both ANNs (see Fig. 3).

% The last step of this part is to design and train 
% two radial basis function (RBF) networks that do the 
% same thing as the previously developed MLPs.

clear
close all
clc

% 1) work with sequentialfs to find features (max 10)

% TODO: PAY ATTENTION
load('C:\Users\gxhan\Documents\Universita\Esami_Correnti\Intelligent Systems\IntelligentSystemsProject\datasets\dataset.mat'); 
X = dataset{:,[1:2 5:end]};

% fun = @(XT,yT,Xt,yt)loss(fitcecoc(XT,yT),Xt,yt);
y_arousal = dataset{:,3};
y_valence = dataset{:,4};

%c = cvpartition(y_arousal,'k', 10);
opts = statset('Display','iter'); 
fun = @(xtrain, ytrain, xtest, ytest) sum(classify(train(perceptron, xtrain, ytrain), xtest) ~= ytest); 
inmodel = sequentialfs(@fun2,X,y_arousal,'options',opts);
disp(inmodel);

% 2) find best architecture for MLPs
mlp_net_arousal = feedforwardnet([10 1]);
mlp_net_valence = feedforwardnet([10 1]);

function err = fun2(xtrain, ytrain, xtest, ytest)
    disp(size(xtrain));
    disp(size(ytrain));
    disp(size(xtest));
    disp(size(ytest));
    net = feedforwardnet([1 3]);
    net = train(net, xtrain, ytrain);
    yret = net(xtest);
    err = sum(1 ~= ytest);
end

