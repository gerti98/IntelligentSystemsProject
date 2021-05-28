%% Preparation of data
clear
close all
clc

%Load datasets
dataset = load('C:\Users\gxhan\Documents\Universita\Esami_Correnti\Intelligent Systems\IntelligentSystemsProject\datasets\dataset.mat'); 
dataset = table2array(dataset.dataset);

%Load results obtained from sequential feature selection
data_prep = load('data_preparation_results_50.mat');

%Data for training
X_train_arousal = data_prep.X_train_best_arousal;
X_train_valence = data_prep.X_train_best_valence;
t_train_arousal = data_prep.t_train_best_arousal;
t_train_valence = data_prep.t_train_best_valence;

%Data for final test of the network to assess performance
X_test_arousal = data_prep.X_test_best_arousal;
X_test_valence = data_prep.X_test_best_valence;
t_test_arousal = data_prep.t_test_best_arousal;
t_test_valence = data_prep.t_test_best_valence;

%% Training MLP FOR AROUSAL
% HYPERPARAMETERS
% ▪ Number of neurons
% ▪ Number of hidden layers
% ▪ Activation functions
% ▪ Learning Rate
% ▪ Momentum
% ▪ Performance function

max_neurons = 50;

for i=1:max_neurons
    mlp_net_arousal = fitnet(max_neurons);
    mlp_net_arousal.divideParam.trainRatio = 0.8; 
    mlp_net_arousal.divideParam.valRatio = 0.2;
    mlp_net_arousal.trainParam.showWindow = 0;
    mlp_net_arousal.trainParam.showCommandLine = 1;
    %mlp_net_arousal.trainParam.mc = 0.1;
    mlp_net_arousal.trainParam.lr = 0.05; %hyper
    mlp_net_arousal.trainParam.epochs = 100;
    mlp_net_arousal.trainParam.max_fail = 25;
    % mlp_net_arousal.trainParam.goal = 0;
    % mlp_net_arousal.trainParam.min_grad = 0;
    [mlp_net_arousal, tr_arousal] = train(mlp_net_arousal, X_train_arousal, t_train_arousal);
    

    y_test_arousal = mlp_net_arousal(X_test_arousal);
    plotregression(t_test_arousal, y_test_arousal, 'Final test arousal');
end
%%  train mlp for valence

max_neurons = 50;

for i=1:max_neurons
    mlp_net_valence = fitnet(max_neurons);
    mlp_net_valence.divideParam.trainRatio = 0.8; 
    mlp_net_valence.divideParam.valRatio = 0.2; 
    mlp_net_valence.divideParam.testRatio = 0;
    mlp_net_valence.trainParam.showWindow = 0;
    mlp_net_valence.trainParam.showCommandLine = 1;
    %mlp_net_valence.trainParam.mc = 0.1;
    mlp_net_valence.trainParam.lr = 0.05; %hyper
    mlp_net_valence.trainParam.epochs = 100;
    mlp_net_valence.trainParam.max_fail = 25;
    % mlp_net_valence.trainParam.goal = 0;
    % mlp_net_valence.trainParam.min_grad = 0;
    [mlp_net_valence, tr_valence] = train(mlp_net_valence, X_train_valence, t_train_valence);
    

    y_test_valence = mlp_net_valence(X_test_valence);
    plotregression(t_test_valence, y_test_valence, 'Final test arousal');
end

%% Part with RBF training for arousal
spread_ar = 0.5;
goal_ar = 0;
K_ar = 600;
Ki_ar = 60;
% create a neural network
rbf_arousal = newrb(X_train_arousal,t_train_arousal,goal,spread, K, Ki);

% Test RBF
y_test_arousal = rbf_arousal(X_test_arousal);
plotregression(t_test_arousal, y_test_arousal, 'Final test arousal');

%% Part with RBF training for valence
spread_va = 1;
goal_va = 0;
K_va = 600;
Ki_va = 60;
rbf_valence = newrb(X_train_valence,t_train_valence,goal_va,spread_va, K_va, Ki_va);

% Test RBF
y_test_valence = rbf_valence(X_test_valence);
plotregression(t_test_valence, y_test_valence, 'Final test arousal');
