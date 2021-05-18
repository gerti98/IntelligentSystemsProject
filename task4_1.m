%% Preparation of data
clear
close all
clc

% Config params
with_dataset_cleaning = 0;
with_minibatch = 0;

%Load datasets
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

num_max_class_arousal = sum(y_arousal(:) == 1);
num_max_class_valence = sum(y_valence(:) == 1);

strange = y_arousal(1);
% Dataset cleaning
if with_dataset_cleaning == 1
    valid_indexes_arousal = [];
    valid_indexes_valence = [];
    samples_arousal = [1 strange 11/3 5 19/3 23/3 9; 0 0 0 0 0 0 0];
    samples_valence = [1 strange 11/3 5 19/3 23/3 9; 0 0 0 0 0 0 0];
    for i = 1:size(y_arousal)
        index = find(samples_arousal(1,:) == y_arousal(i));
        if samples_arousal(2,index) < num_max_class_arousal
            valid_indexes_arousal(end+1) = i;
            samples_arousal(2,index) = samples_arousal(2,index) +1;
        end
    end
    for i = 1:size(y_valence)
        index = find(samples_valence(1,:) == y_valence(i));
        if samples_valence(2,index) < num_max_class_valence
            valid_indexes_valence(end+1) = i;
            samples_valence(2,index) = samples_valence(2,index) +1;
        end
    end

    y_arousal = y_arousal(valid_indexes_arousal)';
    y_valence = y_valence(valid_indexes_valence)';
    Xfeatures_arousal = X(valid_indexes_arousal,features_arousal)';
    Xfeatures_valence = X(valid_indexes_valence,features_valence)';
else
    y_arousal = y_arousal(:)';
    y_valence = y_valence(:)';
    Xfeatures_arousal = X(:,features_arousal)';
    Xfeatures_valence = X(:,features_valence)';
end
%% Training MLP FOR AROUSAL
% HYPERPARAMETERS
% ▪ Number of neurons
% ▪ Number of hidden layers
% ▪ Activation functions
% ▪ Learning Rate
% ▪ Momentum
% ▪ Performance function

% Definition of MLP for arousal
mlp_net_arousal = feedforwardnet([20]);


% Config params
mlp_net_arousal.performFcn = 'msereg';
mlp_net_arousal.divideParam.trainRatio = 0.6; 
mlp_net_arousal.divideParam.valRatio = 0.2; 
mlp_net_arousal.divideParam.testRatio = 0.2; 
mlp_net_arousal.trainFcn = 'trainbr';
mlp_net_arousal.trainParam.lr = 0.1; %hyper
mlp_net_arousal.trainParam.epochs = 1000;
mlp_net_arousal.trainParam.goal = 0;
mlp_net_arousal.trainParam.max_fail = 20;
mlp_net_arousal.trainParam.min_grad = 0;
mlp_net_arousal.trainParam.mc = 0.1;


% For minibatch if needed
if with_minibatch == 1
    batchsize = 24;
    N = size(Xfeatures_arousal, 2); % number of samples
    n_batch = N/batchsize; % number of batches
    input{n_batch} = [ ] ; % input cell −array initialization
    target{n_batch} = [ ] ; % target cell −array initialization 
    p = randperm(N) ; % generating a random permutated index for data shuffling
    Xfeatures_arousal = Xfeatures_arousal( : , p) ; % samples permutation
    y_arousal = y_arousal( : , p) ; % target permutation
    for i = 1 : n_batch
        mini_Xfeatures_arousal{i} = Xfeatures_arousal(:, (1 : batchsize) + (i -1) * batchsize);
        mini_y_arousal{i} = y_arousal(:, (1 : batchsize) + (i - 1) * batchsize);
        [mlp_net_arousal, tr_arousal] = train(mlp_net_arousal, mini_Xfeatures_arousal, mini_y_arousal);
    end
else
    [mlp_net_arousal, tr_arousal] = train(mlp_net_arousal, Xfeatures_arousal, y_arousal);
end

%%  train mlp for valence
mlp_net_valence = feedforwardnet([5 4]);
mlp_net_valence.divideParam.trainRatio = 0.6; 
mlp_net_valence.divideParam.valRatio = 0.2; 
mlp_net_valence.divideParam.testRatio = 0.2; 
mlp_net_valence.trainParam.lr = 0.4; %hyper
mlp_net_valence
% For minibatch if needed
if with_minibatch == 1
    batchsize = 31;
    N = size(Xfeatures_valence, 2); % number of samples
    n_batch = N/batchsize; % number of batches
    input{n_batch} = [ ] ; % input cell −array initialization
    target{n_batch} = [ ] ; % target cell −array initialization 
    p = randperm(N) ; % generating a random permutated index for data shuffling
    Xfeatures_valence = Xfeatures_valence( : , p) ; % samples permutation
    y_valence = y_valence( : , p) ; % target permutation
    for i = 1 : n_batch
        mini_Xfeatures_valence{i} = Xfeatures_valence(:, (1 : batchsize) + (i -1) * batchsize);
        mini_y_valence{i} = y_valence(:, (1 : batchsize) + (i - 1) * batchsize);
        [mlp_net_valence, tr_valence] = train(mlp_net_valence, mini_Xfeatures_valence, mini_y_valence);
    end
else
    mlp_net_valence = train(mlp_net_valence, Xfeatures_valence, y_valence);
end


%% Test the Network
% The mean squared error of the trained neural network can now be measured with respect to the testing samples.
% This will give us a sense of how well the network will do when applied to data from the real world.
% The network outputs will be in the range 0 to 1,
% so we can use vec2ind function to get the class indices as the position
% of the highest element in each output vector.
% testX_arousal = Xfeatures_arousal(:, tr_arousal.testInd);
% testT_arousal = y_arousal(:, tr_arousal.testInd);
% 
% testY_arousal = mlp_net_arousal(testX_arousal);
% testIndices = vec2ind(testY_arousal);


% Another measure of how well the neural network has fit the data is the confusion plot.
% The confusion matrix shows the percentages of correct and incorrect classifications.
% Correct classifications are the green squares on the matrices diagonal.
% Incorrect classifications form the red squares.

% If the network has learned to classify properly,
% the percentages in the red squares should be very small, indicating few misclassifications.
% plotconfusion(testT_arousal, testY_arousal);
% hold on;
% 
% Here are the overall percentages of correct and incorrect classification.
% [c, cm] = confusion(testT_arousal, testY_arousal);
% fprintf('Percentage Correct Classification: %f%%\n', 100*(1-c));
% fprintf('Percentage Incorrect Classification: %f%%\n', 100*c);

% A third measure of how well the neural network has fit data is the receiver operating characteristic plot.
% This shows how the false positive and true positive rates relate as the thresholding of outputs is varied from 0 to 1.
% The farther left and up the line is, the fewer false positives need to be accepted
% in order to get a high true positive rate.
% The best classifiers will have a line going from the bottom left corner,
% to the top left corner, to the top right corner, or close to that.
% plotroc(testT_arousal, testY_arousal);


%% Part with RBF training
%---------------------------------
% choose a spread constant
spread = .2;
% choose max number of neurons
K = 100;
% performance goal (SSE)
goal = 0;
% number of neurons to add between displays
Ki = 2;
% create a neural network
rbf_arousal = newrb(Xfeatures_arousal,y_arousal,goal,spread,K,Ki);

% % simulate a network over complete input range
Y = rbf_arousal(Xfeatures_arousal);
% plot network response
fig = figure;
figure(fig)
plot(Y,y_arousal,'k+');

% xlabel('x');
% ylabel('y');
% legend('original function','available data','RBFN','location','northwest')


