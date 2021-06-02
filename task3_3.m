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

nbins = 15;
binWidth = 0.5;

x_values_37 = X_train_arousal(1, :);
% x_values_1_sorted = sort(x_values_37);
x_values_51 = X_train_arousal(2, :);
% x_values_2_sorted = sort(x_values_51);
x_values_12 = X_train_arousal(3, :);
% x_values_3_sorted = sort(x_values_12);
y_values = t_train_arousal;


y_indexes_one = find(y_values == 1);
y_indexes_two = find(y_values == y_values(10));
y_indexes_three = find(y_values == 11/3);
y_indexes_four = find(y_values == 5);
y_indexes_five = find(y_values == 19/3);
y_indexes_six = find(y_values == 23/3);
y_indexes_seven = find(y_values == 9);

% figure
% t = tiledlayout(7,3);
% 
% nexttile
% histogram(x_values_37(y_indexes_one),'BinWidth',binWidth);
% title('Histogram of feature 1 when y: 1');
% nexttile
% histogram(x_values_51(y_indexes_one),'BinWidth',binWidth);
% title('Histogram of feature 2 when y: 1');
% nexttile
% histogram(x_values_12(y_indexes_one),'BinWidth',binWidth);
% title('Histogram of feature 3 when y: 1');
% nexttile
% 
% histogram(x_values_37(y_indexes_two),'BinWidth',binWidth);
% title('Histogram of feature 1 when y: 2.333');
% nexttile
% histogram(x_values_51(y_indexes_two),'BinWidth',binWidth);
% title('Histogram of feature 2 when y: 2.333');
% nexttile
% histogram(x_values_12(y_indexes_two),'BinWidth',binWidth);
% title('Histogram of feature 3 when y: 2.333');
% nexttile
% 
% histogram(x_values_37(y_indexes_three),'BinWidth',binWidth);
% title('Histogram of feature 1 when y: 3.667');
% nexttile
% histogram(x_values_51(y_indexes_three),'BinWidth',binWidth);
% title('Histogram of feature 2 when y: 3.667');
% nexttile
% histogram(x_values_12(y_indexes_three),'BinWidth',binWidth);
% title('Histogram of feature 3 when y: 3.667');
% nexttile
% 
% histogram(x_values_37(y_indexes_four),'BinWidth',binWidth);
% title('Histogram of feature 1 when y: 5');
% nexttile
% histogram(x_values_51(y_indexes_four),'BinWidth',binWidth);
% title('Histogram of feature 2 when y: 5');
% nexttile
% histogram(x_values_12(y_indexes_four),'BinWidth',binWidth);
% title('Histogram of feature 3 when y: 5');
% nexttile
% 
% histogram(x_values_37(y_indexes_five),'BinWidth',binWidth);
% title('Histogram of feature 1 when y: 6.333');
% nexttile
% histogram(x_values_51(y_indexes_five),'BinWidth',binWidth);
% title('Histogram of feature 2 when y: 6.333');
% nexttile
% histogram(x_values_12(y_indexes_five),'BinWidth',binWidth);
% title('Histogram of feature 3 when y: 6.333');
% nexttile
% 
% histogram(x_values_37(y_indexes_six),'BinWidth',binWidth);
% title('Histogram of feature 1 when y: 7.667');
% nexttile
% histogram(x_values_51(y_indexes_six),'BinWidth',binWidth);
% title('Histogram of feature 2 when y: 7.667');
% nexttile
% histogram(x_values_12(y_indexes_six),'BinWidth',binWidth);
% title('Histogram of feature 3 when y: 7.667');
% nexttile
% 
% histogram(x_values_37(y_indexes_seven),'BinWidth',binWidth);
% title('Histogram of feature 1 when y: 9');
% nexttile
% histogram(x_values_51(y_indexes_seven),'BinWidth',binWidth);
% title('Histogram of feature 2 when y: 9');
% nexttile
% histogram(x_values_12(y_indexes_seven),'BinWidth',binWidth);
% title('Histogram of feature 3 when y: 9');
% 


figure
y_lim = 15;

t = tiledlayout(3,3);
low_ind = [y_indexes_one y_indexes_two y_indexes_three];
mid_ind = [y_indexes_three y_indexes_four y_indexes_five];
high_ind = [y_indexes_five y_indexes_six y_indexes_seven];

nexttile
histogram(x_values_37(low_ind), 'BinWidth',binWidth, 'BinLimits',[-4,10]);
yline(y_lim, '--r');
title('Histogram of feature 37 when y: low [1-3)');
nexttile
histogram(x_values_51(low_ind),'BinWidth',binWidth, 'BinLimits',[-4,10]);
title('Histogram of feature 51 when y: low [1-3)');
yline(y_lim, '--r');
nexttile
histogram(x_values_12(low_ind),'BinWidth',binWidth, 'BinLimits',[-4,10]);
yline(y_lim, '--r');
title('Histogram of feature 12 when y: low [1-3)');

nexttile
histogram(x_values_37(mid_ind),'BinWidth',binWidth, 'BinLimits',[-4,10]);
yline(y_lim, '--r');
title('Histogram of feature 37 when y: mid [3-7]');
nexttile
histogram(x_values_51(mid_ind),'BinWidth',binWidth, 'BinLimits',[-4,10]);
yline(y_lim, '--r');
title('Histogram of feature 51 when y: mid [3-7]');
nexttile
histogram(x_values_12(mid_ind),'BinWidth',binWidth, 'BinLimits',[-4,10]);
yline(y_lim, '--r');
title('Histogram of feature 12 when y: mid [3-7]');

nexttile
histogram(x_values_37(high_ind),'BinWidth',binWidth, 'BinLimits',[-4,10]);
yline(y_lim, '--r');
title('Histogram of feature 37 when y: high (7-9]');
nexttile
histogram(x_values_51(high_ind), 'BinWidth',binWidth, 'BinLimits',[-4,10]);
yline(y_lim, '--r');
title('Histogram of feature 51 when y: high (7-9]');
nexttile
histogram(x_values_12(high_ind), 'BinWidth',binWidth, 'BinLimits',[-4,10]);
yline(y_lim, '--r');
title('Histogram of feature 12 when y: high (7-9]');

% Plot correlations

figure
t = tiledlayout(1,3);
nexttile
scatter(x_values_37, x_values_51);
title('Scatterplot of feature 37 and feature 51');
nexttile
scatter(x_values_37, x_values_12);
title('Scatterplot of feature 37 and feature 12');
nexttile
scatter(x_values_51, x_values_12);
title('Scatterplot of feature 51 and feature 12');

figure
t = tiledlayout(1,3);
nexttile
histogram(x_values_37,'BinWidth',binWidth);
title('Histogram of feature 37');
nexttile
histogram(x_values_51, 'BinWidth',binWidth);
title('Histogram of feature 51');
nexttile
histogram(x_values_12, 'BinWidth',binWidth);
title('Histogram of feature 12');

% corr_1 = [x_values_37; y_values]';
% corr_1 = sortrows(corr_1, 1, 'ascend');
% corr_2 = [x_values_51; y_values]';
% corr_2 = sortrows(corr_2, 1, 'ascend');
% corr_3 = [x_values_12; y_values]';
% corr_3 = sortrows(corr_3, 1, 'ascend');


%% Fuzzy logic systems
fis = mamfis("Name", "MamdaniFis");

fis = addInput(fis,[min(x_values_37) max(x_values_37)],'Name', string(best3_features(1)));
fis = addMF(fis,string(best3_features(1)),'trapmf',[-10 -3 0 2],'Name',"Low");
fis = addMF(fis,string(best3_features(1)),'trimf',[0 2 4],'Name',"Medium");
fis = addMF(fis,string(best3_features(1)),'trapmf',[2 4 10 13],'Name',"High");

fis = addInput(fis,[min(x_values_51) max(x_values_51)],'Name', string(best3_features(2)));
fis = addMF(fis,string(best3_features(2)),'trapmf',[-8.38 -4 -1 1],'Name',"Low");
fis = addMF(fis,string(best3_features(2)),'trimf',[-1 1 3],'Name',"Medium");
fis = addMF(fis,string(best3_features(2)),'trapmf',[1 3 10 11],'Name',"High");

fis = addInput(fis,[min(x_values_12) max(x_values_12)],'Name', string(best3_features(3)));
fis = addMF(fis,string(best3_features(3)),'trapmf',[-11.59 -3 3 6],'Name',"Low");
fis = addMF(fis,string(best3_features(3)),'trapmf',[3 6 12 18.7],'Name',"High");

fis = addOutput(fis,[1 9],'Name',"Arousal");
fis = addMF(fis,"Arousal","trimf",[-3 1 5],'Name',"Low");
fis = addMF(fis,"Arousal","trimf",[1 5 9],'Name',"Medium");
fis = addMF(fis,"Arousal","trimf",[5 9 13],'Name',"High");

ruleList = [1 1 0 1 1 1;
            2 1 0 1 1 1;
            0 0 1 1 1 1;
            2 1 0 2 1 1;
            3 1 0 2 1 1;
            2 2 0 3 1 1;
            3 3 0 3 1 1;
            0 0 2 3 1 1;
            ];

fis = addRule(fis,ruleList);
evalfis(fis,[1 1 2])