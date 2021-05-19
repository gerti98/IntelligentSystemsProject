%% Preparation of data
clear
close all
clc

t = ones(1,1112);
imds = imageDatastore('datasets/images/anger_to_use', 'labels', t);
% Split the data into training and testing sets
fracTrain = 0.8;
[imdsTrain,imdsTest] = splitEachLabel(imds,fracTrain,'randomize');


base_img = imread('datasets/images/anger_to_use/10011.jpg');

layers = [
    imageInputLayer(size(base_img))
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2) 
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2) 
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];


%% Training 
% The mini-batch size should be less than the data set size; the mini-batch 
% is used at each training iteration to evaluate gradients and update the 
% weights. 
options = trainingOptions('sgdm', ...
'InitialLearnRate', 0.01, ...
'MiniBatchSize', 16, ...
'MaxEpochs', 5, ...
'Shuffle','every-epoch', ...
'ValidationData', imdsTest, ...
'ValidationFrequency', 2, ...
'Verbose', false, ...
'Plots', 'training-progress');

net = trainNetwork(imdsTrain,layers,options);