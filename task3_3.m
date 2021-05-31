%% Preparation of data
clear
close all
clc

%Load datasets
dataset_ = load('C:\Users\gxhan\Documents\Universita\Esami_Correnti\Intelligent Systems\IntelligentSystemsProject\datasets\dataset.mat'); 
dataset = table2array(dataset_.dataset);

%Load results obtained from sequential feature selection
data_prep = load('data_preparation_results_100.mat');

best3_features = data_prep.features_arousal(1:3,2);
features_names = dataset_.dataset.Properties.VariableNames(5:58);
best3_features_names = features_names(best3_features);

%Data for training
X_train_arousal = data_prep.X_train_best3_arousal;
t_train_arousal = data_prep.t_train_best_arousal;

%Data for final test of the network to assess performance
X_test_arousal = data_prep.X_test_best3_arousal;
t_test_arousal = data_prep.t_test_best_arousal;

%% Observation of data

x_values_1 = X_train_arousal(1, :);
x_values_1_sorted = sort(x_values_1);
x_values_2 = X_train_arousal(2, :);
x_values_2_sorted = sort(x_values_2);
x_values_3 = X_train_arousal(3, :);
x_values_3_sorted = sort(x_values_3);
y_values = t_train_arousal;

corr_1 = [x_values_1; y_values]';
corr_1 = sortrows(corr_1, 1, 'ascend');
corr_2 = [x_values_2; y_values]';
corr_2 = sortrows(corr_2, 1, 'ascend');
corr_3 = [x_values_3; y_values]';
corr_3 = sortrows(corr_3, 1, 'ascend');

%% Fuzzy logic systems
fis = mamfis("Name", "MamdaniFis");

fis = addInput(fis,[min(x_values_1) max(x_values_1)],'Name', string(best3_features_names(1)));
fis = addMF(fis,string(best3_features_names(1)),"trimf",[-6.265 -2.598 1.069],'Name',"Very Low");
fis = addMF(fis,string(best3_features_names(1)),"trimf",[-2.598 1.069 4.736],'Name',"Low");
fis = addMF(fis,string(best3_features_names(1)),"trimf",[1.069 4.736 8.403],'Name',"High");
fis = addMF(fis,string(best3_features_names(1)),"trimf",[4.736 8.403 12.07],'Name',"Very High");

fis = addInput(fis,[min(x_values_2) max(x_values_2)],'Name', string(best3_features_names(2)));
fis = addMF(fis,string(best3_features_names(2)),"trimf",[-7.032 -3.181 0.6693],'Name',"Very Low");
fis = addMF(fis,string(best3_features_names(2)),"trimf",[-3.181 0.6693 4.52],'Name',"Low");
fis = addMF(fis,string(best3_features_names(2)),"trimf",[0.6693 4.52 8.371],'Name',"High");
fis = addMF(fis,string(best3_features_names(2)),"trimf",[4.52 8.371 12.22],'Name',"Very High");

fis = addInput(fis,[min(x_values_3) max(x_values_3)],'Name', string(best3_features_names(3)));
fis = addMF(fis,string(best3_features_names(3)),"trimf",[-5.462 -1.855 1.752],'Name',"Very Low");
fis = addMF(fis,string(best3_features_names(3)),"trimf",[-1.855 1.752 5.358],'Name',"Low");
fis = addMF(fis,string(best3_features_names(3)),"trimf",[1.752 5.358 8.965],'Name',"High");
fis = addMF(fis,string(best3_features_names(3)),"trimf",[5.358 8.965 12.57],'Name',"Very High");

fis = addOutput(fis,[1 9],'Name',"Arousal");
fis = addMF(fis,"Arousal","trimf",[-0.3333 1 2.333],'Name',"Very Very Low");
fis = addMF(fis,"Arousal","trimf",[1 2.333 3.667],'Name',"Very Low");
fis = addMF(fis,"Arousal","trimf",[2.333 3.667 5],'Name',"Low");
fis = addMF(fis,"Arousal","trimf",[3.667 5 6.333],'Name',"Medium");
fis = addMF(fis,"Arousal","trimf",[5 6.333 7.667],'Name',"High");
fis = addMF(fis,"Arousal","trimf",[6.333 7.667 9],'Name',"Very High");
fis = addMF(fis,"Arousal","trimf",[7.667 9 10.33],'Name',"Very Very High");

ruleList = [1 0 0 1 1 1;];

fis = addRule(fis,ruleList);
evalfis(fis,[1 1 2])