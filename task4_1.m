%% Preparation of data
clear
close all
clc

dataset = load('C:\Users\gxhan\Documents\Universita\Esami_Correnti\Intelligent Systems\IntelligentSystemsProject\datasets\dataset.mat'); 
dataset = table2array(dataset.dataset);

%Load results obtained from sequential feature selection
features_arousal_sorted = load('features_arousal_sorted.mat');
features_arousal_sorted = features_arousal_sorted.features_arousal;
features_valence_sorted = load('features_valence_sorted.mat');
features_valence_sorted = features_valence_sorted.features_valence;


X = dataset(:,5:end);
y_arousal = dataset(:,3);
y_valence = dataset(:,4);

%Load best 10 features extracted from sequentialfs for both arousal and
%valence
features_arousal = features_arousal_sorted(1:10, 2);
features_valence = features_valence_sorted(1:10, 2);


%% Training MLP
% HYPERPARAMETERS
% ▪ Number of neurons
% ▪ Number of hidden layers
% ▪ Activation functions
% ▪ Learning Rate
% ▪ Momentum
% ▪ Performance function

Xfeatures_arousal = X(:,features_arousal)';
Xfeatures_valence = X(:,features_valence)';
y_arousal = y_arousal';
y_valence = y_valence';

% counterexample - general case
% mlp_net_arousal_full = feedforwardnet([10 5]);
% view(mlp_net_arousal_full);
% 
% mlp_net_arousal_full.divideParam.trainRatio = 0.7; 
% mlp_net_arousal_full.divideParam.valRatio = 0.2; 
% mlp_net_arousal_full.divideParam.testRatio = 0.1; 
% mlp_net_arousal = train(mlp_net_arousal_full, X', y_arousal);

% train mlp for arousal
mlp_net_arousal = feedforwardnet([5 4]);
% view(mlp_net_arousal);


mlp_net_arousal.divideParam.trainRatio = 0.6; 
mlp_net_arousal.divideParam.valRatio = 0.2; 
mlp_net_arousal.divideParam.testRatio = 0.2; 
mlp_net_arousal.trainParam.lr = 0.1; %hyper
% mlp_net_arousal.transferFcn = 'logsig';
[mlp_net_arousal, tr_arousal] = train(mlp_net_arousal, Xfeatures_arousal, y_arousal);

% train mlp for valence
% mlp_net_valence = feedforwardnet([5 3]);
% view(mlp_net_valence);
% 
% mlp_net_valence.divideParam.trainRatio = 0.6; 
% mlp_net_valence.divideParam.valRatio = 0.2; 
% mlp_net_valence.divideParam.testRatio = 0.2; 
% mlp_net_valence = train(mlp_net_valence, Xfeatures_valence, y_valence);


%% Test the Network
% The mean squared error of the trained neural network can now be measured with respect to the testing samples.
% This will give us a sense of how well the network will do when applied to data from the real world.
% The network outputs will be in the range 0 to 1,
% so we can use vec2ind function to get the class indices as the position
% of the highest element in each output vector.
testX_arousal = Xfeatures_arousal(:, tr_arousal.testInd);
testT_arousal = y_arousal(:, tr_arousal.testInd);

testY_arousal = mlp_net_arousal(testX_arousal);
testIndices = vec2ind(testY_arousal);


% Another measure of how well the neural network has fit the data is the confusion plot.
% The confusion matrix shows the percentages of correct and incorrect classifications.
% Correct classifications are the green squares on the matrices diagonal.
% Incorrect classifications form the red squares.

% If the network has learned to classify properly,
% the percentages in the red squares should be very small, indicating few misclassifications.
plotconfusion(testT_arousal, testY_arousal);
hold on;
% 
% Here are the overall percentages of correct and incorrect classification.
[c, cm] = confusion(testT_arousal, testY_arousal);
fprintf('Percentage Correct Classification: %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification: %f%%\n', 100*c);

% A third measure of how well the neural network has fit data is the receiver operating characteristic plot.
% This shows how the false positive and true positive rates relate as the thresholding of outputs is varied from 0 to 1.
% The farther left and up the line is, the fewer false positives need to be accepted
% in order to get a high true positive rate.
% The best classifiers will have a line going from the bottom left corner,
% to the top left corner, to the top right corner, or close to that.
% plotroc(testT_arousal, testY_arousal);
