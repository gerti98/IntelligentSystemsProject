%% Preparation of data
clear
close all
clc

anger_cell = cell(1,250);
anger_cell(:) = {'Anger'};
disgust_cell = cell(1,250);
disgust_cell(:) = {'Disgust'};
fear_cell = cell(1,250);
fear_cell(:) = {'Fear'};
happiness_cell = cell(1,250);
happiness_cell(:) = {'Happiness'};
base_img = imread('datasets/images/anger_to_use/10011.jpg');
base_img_size = size(base_img);


emotions_labels = categorical([anger_cell disgust_cell fear_cell happiness_cell]);
imds = imageDatastore('datasets/images/images_to_use_experiment', 'labels', emotions_labels);
% Split the data into training and testing sets
fracTrain = 0.8;
[imdsTrain,imdsTest] = splitEachLabel(imds,fracTrain,'randomize');


imageAugmenter = imageDataAugmenter('RandRotation',[-20,20]);
augimds = augmentedImageDatastore(base_img_size,imdsTrain,'DataAugmentation',imageAugmenter);


% layers = [
%     imageInputLayer(size(base_img))
%     
%     convolution2dLayer(10,24,'Stride',4,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(5,48,'Stride',2,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,96,'Stride',1,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,96,'Stride',1,'Padding','same')
%     batchNormalizationLayer
%     leakyReluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,96,'Stride',1,'Padding','same')
%     batchNormalizationLayer
%     leakyReluLayer
%     
%     fullyConnectedLayer(100)
%     fullyConnectedLayer(4)
%     
%     softmaxLayer
%     
%     classificationLayer
% ];

leaky_layers = [
    imageInputLayer(size(base_img))
    
    convolution2dLayer(16,12,'Stride',4,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(8,24,'Stride',2,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(4,48,'Stride',1,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    
    fullyConnectedLayer(100)
    fullyConnectedLayer(100)
    fullyConnectedLayer(4)
    
    softmaxLayer
    
    classificationLayer
];



%% Training

options_sgdm = trainingOptions('sgdm', ...
'InitialLearnRate', 0.01, ...
'MiniBatchSize', 40, ...
'MaxEpochs', 40, ...
'Shuffle','every-epoch', ...
'ValidationData', imdsTest, ...
'ValidationFrequency', 20, ...
'Verbose', false, ...
'Plots', 'training-progress');

% options_adam = trainingOptions('adam', ...
% 'InitialLearnRate', 0.01, ...
% 'MiniBatchSize', 20, ...
% 'MaxEpochs', 40, ...
% 'Shuffle','every-epoch', ...
% 'ValidationData', imdsTest, ...
% 'ValidationFrequency', 4, ...
% 'Verbose', false, ...
% 'Plots', 'training-progress');
% 
% 
% options_rmsprop = trainingOptions('rmsprop', ...
% 'InitialLearnRate', 0.01, ...
% 'MiniBatchSize', 20, ...
% 'MaxEpochs', 40, ...
% 'Shuffle','every-epoch', ...
% 'ValidationData', imdsTest, ...
% 'ValidationFrequency', 20, ...
% 'Verbose', false, ...
% 'Plots', 'training-progress');

% net_sgdm = trainNetwork(augimds,layers,options_sgdm);
% net_adam = trainNetwork(augimds,layers,options_adam);
% net_rmsprop = trainNetwork(augimds,layers,options_rmsprop);
net_sgdm_leaky = trainNetwork(augimds,leaky_layers,options_sgdm);
% net_adam_leaky = trainNetwork(augimds,leaky_layers,options_adam);
% net_rmsprop_leaky = trainNetwork(augimds,leaky_layers,options_rmsprop);

% predLabels_sgdm = classify(net_sgdm,imdsTest);
% predLabels_adam = classify(net_adam,imdsTest);
% predLabels_rmsprop = classify(net_rmsprop,imdsTest);
predLabels_sgdm_leaky = classify(net_sgdm_leaky,imdsTest);
% predLabels_adam_leaky = classify(net_adam_leaky,imdsTest);
% predLabels_rmsprop_leaky = classify(net_rmsprop_leaky,imdsTest);

testLabels = imdsTest.Labels;
% accuracy_sgdm = sum(predLabels_sgdm == testLabels)/numel(testLabels);
% fprintf('Accuracy sgdm is %8.2f%%\n',accuracy_sgdm*100);
% accuracy_adam = sum(predLabels_adam == testLabels)/numel(testLabels);
% fprintf('Accuracy adam is %8.2f%%\n',accuracy_adam*100);
% accuracy_rmsprop = sum(predLabels_rmsprop == testLabels)/numel(testLabels);
% fprintf('Accuracy rmsprop is %8.2f%%\n',accuracy_rmsprop*100);
accuracy_sgdm_leaky = sum(predLabels_sgdm_leaky == testLabels)/numel(testLabels);
fprintf('Accuracy sgdm leaky is %8.2f%%\n',accuracy_sgdm_leaky*100);
% accuracy_adam_leaky = sum(predLabels_adam_leaky == testLabels)/numel(testLabels);
% fprintf('Accuracy adam leaky is %8.2f%%\n',accuracy_adam_leaky*100);
% accuracy_rmsprop_leaky = sum(predLabels_rmsprop_leaky == testLabels)/numel(testLabels);
% fprintf('Accuracy rmsprop leaky is %8.2f%%\n',accuracy_rmsprop_leaky*100);



