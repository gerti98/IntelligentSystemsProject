%% Configuration
clear
close all
clc

times_sequentialfs = 15;
outliers_removal_method = 'median';

%% Preparation of Data


% Removal of non numerical values
dataset = load('C:\Users\gxhan\Documents\Universita\Esami_Correnti\Intelligent Systems\IntelligentSystemsProject\datasets\dataset.mat'); 
dataset = table2array(dataset.dataset);
inf_val = isinf(dataset);
[rows_inf, col_inf] = find(inf_val == 1);
dataset(rows_inf,:) = [];

% Removal of outliers
dataset = dataset(:, 3:end);
dataset = rmoutliers(dataset, outliers_removal_method);


X = dataset(:,3:end);
t_arousal = dataset(:,1);
t_valence = dataset(:,2);


% Dataset balancing: balancing of indexes found through a trial and error
% approach
index = 1;
while index < 25
    %count of number of samples with specific output 
    [GC_arousal,GR_arousal] = groupcounts(t_arousal);
    [GC_valence,GR_valence] = groupcounts(t_valence);
    
    [M,I] = max(GC_arousal);
    [m,i] = min(GC_arousal);
    max_val_ar = GR_arousal(I);
    min_val_ar = GR_arousal(i);
    min_val_indexes_ar = find(t_arousal == min_val_ar);
    max_val_indexes_ar = find(t_arousal == max_val_ar);
    
    
    [M2, I2] = max(GC_valence);
    [m2, i2] = min(GC_valence);
    max_val_va = GR_valence(I2);
    min_val_va = GR_valence(i2);
    min_val_indexes_va = find(t_valence == min_val_va);
    max_val_indexes_va = find(t_valence == max_val_va);
    
    % In order to not increase the most common class of valence(arousal)
    % the relative indexes are deleted from the process in case of data 
    % augmentation of least common class of arousal(valence) 
    min_val_indexes_ar_common = intersect(min_val_indexes_ar,max_val_indexes_va);
    min_val_indexes_ar = setxor(min_val_indexes_ar,min_val_indexes_ar_common);

    min_val_indexes_va_common = intersect(min_val_indexes_va,max_val_indexes_ar);
    min_val_indexes_va = setxor(min_val_indexes_va,min_val_indexes_va_common);
    
    % Limit data augmentation of the current iteration in order to not
    % overcome the most common class of the same type
    if(size(min_val_indexes_ar, 1) > (M-m))
        diff = M-m;
        min_val_indexes_ar = min_val_indexes_ar(1:diff); 
    end
    if(size(min_val_indexes_va, 1) > (M2-m2))
        diff = M2-m2;
        min_val_indexes_va = min_val_indexes_va(1:(M2-m2));
    end
    
    % Random value between 0.95 and 1.05 for data augmentation, the
    % selected data will be multiplied by the randomizer
    randomizer = 1 + (rand(1) - 0.5)/10;
    
    if (index <= 5) | ((index >= 16) & (index <=19))
        t_valence = [t_valence; t_valence(min_val_indexes_va)];
        t_arousal = [t_arousal; t_arousal(min_val_indexes_va)];
        X = [X; randomizer*X(min_val_indexes_va, :)];
    elseif ((index >= 6) & (index <=15)) | (index >= 20)
        t_arousal = [t_arousal; t_arousal(min_val_indexes_ar)];
        t_valence = [t_valence; t_valence(min_val_indexes_ar)];
        X = [X; randomizer*X(min_val_indexes_ar, :)];
    end    
    index = index + 1;
end

tiledlayout(2, 1);
nexttile;
bar(GC_arousal);
title('Samples per arousal class');
nexttile;
bar(GC_valence,'y');
title('Samples per valence class');

%% Feature Selection

% Firstly an holdout partition is created in order to no bias the feature
% selection process with the test set created
p = 0.30;

c = cvpartition(t_arousal,'Holdout',p);
idxTrain = training(c);
X_train = X(idxTrain, :);
t_arousal_train = t_arousal(idxTrain);
t_valence_train = t_valence(idxTrain);

idxTest = test(c);
X_test = X(idxTest, :);
t_arousal_test = t_arousal(idxTest);
t_valence_test = t_valence(idxTest);

% Matrix to count how many times a specific feature has been selected by
% sequentialfs, at the end will be sorted in order to determine the best
% features
features_arousal = [zeros(1,54); 1:54]';
features_valence = [zeros(1,54); 1:54]';

%% Feature selection for arousal
for i = 1:times_sequentialfs
    disp("**** ITER ****");
    disp(i);

    cv = cvpartition(t_arousal_train, 'k', 10);
    opt = statset('display','iter','useParallel',true);
    inmodel = sequentialfs(@fun2,X_train,t_arousal_train,'cv', cv,'Options', opt);
    
    % Fetch useful indexes from result of latter sequentialfs
    i = 1;
    for val = inmodel
        if val == 1
            features_arousal(i, 1) = features_arousal(i, 1) + 1;
            fprintf("Added %d\n",i);
        end
        i = i+1;
    end
    
    % features_arousal = unique(features_arousal);
    
    disp("*** Features: ");
    disp(features_arousal);
end
fprintf("**********************************\n");
fprintf("*** AROUSAL: "); 
disp(features_arousal);
fprintf("**********************************\n");

disp("Sorting...");
features_arousal = sortrows(features_arousal, 1, 'descend');
disp(features_arousal);

%% Feature selection for valence
for i = 1:times_sequentialfs
    disp("**** ITER ****");
    disp(i);

    cv = cvpartition(t_valence_train, 'k', 10);
    opt = statset('display','iter','useParallel',true);
    inmodel = sequentialfs(@fun2,X_train,t_valence_train,'cv', cv,'Options', opt);
    
    % Fetch useful indexes from result of latter sequentialfs
    i = 1;
    for val = inmodel
        if val == 1
            features_valence(i, 1) = features_valence(i, 1) + 1;
            fprintf("Added %d\n",i);
        end
        i = i+1;
    end
    
    % features_arousal = unique(features_arousal);
    
    disp("*** Features: ");
    disp(features_valence);
end
fprintf("**********************************\n");
fprintf("*** AROUSAL: "); 
disp(features_valence);
fprintf("**********************************\n");

disp("Sorting...");
features_valence = sortrows(features_valence, 1, 'descend');
disp(features_valence);
 
%% Preparing the outputs for other tasks 

%Select best 10 features for both arousal and valence
features_arousal_best = features_arousal(1:10, 2);
features_valence_best = features_valence(1:10, 2);
features_arousal_best3 = features_arousal(1:3, 2);

X_train_best_arousal = X_train(:,features_arousal_best)';
X_train_best_valence = X_train(:,features_valence_best)';
X_test_best_arousal = X_test(:,features_arousal_best)';
X_test_best_valence = X_test(:,features_valence_best)';
t_train_best_arousal = t_arousal_train';
t_train_best_valence = t_valence_train';
t_test_best_arousal = t_arousal_test';
t_test_best_valence = t_valence_test';

%For task 3.3
X_train_best3_arousal = X_train(:,features_arousal_best3)';
X_test_best3_arousal = X_test(:,features_arousal_best3)';


%% Function for sequentialfs
function err = fun2(x_train, t_train, x_test, t_test)
    %since 1038 samples => 1 output and 1 input (2 neurons) N_hidden = Nsamples/(10*(Ninput+Noutput)) ~= 60 
    
    net = fitnet(60);
    net.trainParam.showWindow=0;
    xx = x_train';
    tt = t_train';
    net = train(net, xx, tt);
    
    % test the network 
    y=net(x_test'); 
    err = perform(net,t_test',y);
end