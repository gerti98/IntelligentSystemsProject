%% Preparation of data
clear
close all
clc

dataset = load('C:\Users\gxhan\Documents\Universita\Esami_Correnti\Intelligent Systems\IntelligentSystemsProject\datasets\dataset.mat'); 
dataset = table2array(dataset.dataset);
X = dataset(:,5:end);
y_arousal = dataset(:,3);
y_valence = dataset(:,4);

features_arousal = [zeros(1,54); 1:54]';
features_valence = [zeros(1,54); 1:54]';

features_demo_arousal = [27 30 31 35 40 41 42 43 44 47]; %TODO just for test
features_demo_valence = [10 27 30 35 40 42 43 47 49 50]; %TODO just for test

%% Feature selection for arousal
for i = 1:100
    ncol = 10;
    x = randperm(size(X, 2),ncol);
    columns = X(:,x);
    % disp(x);
    opt = statset('display','iter','useParallel',true);
    inmodel = sequentialfs(@fun2,columns,y_arousal, 'Options', opt);
    
    
    % Fetch useful indexes from result of latter sequentialfs
    i = 1;
    for val = inmodel
        if val == 1
            features_arousal(x(i), 1) = features_arousal(x(i), 1) + 1;
            fprintf("Added %d\n",x(i)); 
        end
        i = i+1;
    end
    
    % features_arousal = unique(features_arousal);
    
    disp("*** Features: ");
    disp(features_arousal);
%     if length(features_arousal) >= 10
%         break
%     end
end
fprintf("**********************************\n");
fprintf("*** AROUSAL: "); 
disp(features_arousal);
fprintf("**********************************\n");

disp("Sorting...");
features_arousal = sortrows(features_arousal, 1, 'descend');
disp(features_arousal);


%% Feature selection for valence
for i = 1:100
    disp("**** ITER ****");
    disp(i);
    ncol = 10;
    x = randperm(size(X, 2),ncol);
    columns = X(:,x);
    disp(x);
    opt = statset('display','iter', 'useParallel', true);
    inmodel = sequentialfs(@fun2,columns,y_arousal, 'Options', opt);
    
    % Fetch useful indexes from result of latter sequentialfs
    i = 1;
    for val = inmodel
        if val == 1
            features_valence(x(i), 1) = features_valence(x(i), 1) + 1;
            fprintf("Added %d\n",x(i)); 
        end
        i = i+1;
    end
  
%     features_valence = unique(features_valence);
    disp("*** Features: ");
    disp(features_valence);
%     if length(features_valence) >= 10
%         break
%     end
end

fprintf("**********************************\n");
fprintf("*** VALENCE: ");
disp(features_valence);
fprintf("**********************************\n");
disp("Sorting...");
features_valence = sortrows(features_valence, 1, 'descend');
disp(features_valence);



%% Function for sequentialfs
function err = fun2(xtrain, ytrain, xtest, ytest)
    net = fitnet(10);
    net.trainParam.showWindow=0;
    net = train(net, xtrain', ytrain');
    ttest = net(xtest');
    err = mse(fitnet(10), ttest, ytest');
end