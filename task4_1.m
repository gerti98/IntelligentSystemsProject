%% Preparation of data
clear
close all
clc

% anger_cell = cell(1,1000);
% anger_cell(:) = {'Anger'};
% disgust_cell = cell(1,1000);
% disgust_cell(:) = {'Disgust'};
% fear_cell = cell(1,1000);
% fear_cell(:) = {'Fear'};
% happiness_cell = cell(1,1000);
% happiness_cell(:) = {'Happiness'};
% 
% emotions_labels = categorical([anger_cell disgust_cell fear_cell happiness_cell]);
% imds = imageDatastore('datasets/images/images_to_use', 'labels', emotions_labels);

% % Split the data into training and testing sets
% fracTrain = 0.8;
% [imdsTrain,imdsTest] = splitEachLabel(imds,fracTrain,'randomize');

anger_cell = cell(1,50);
anger_cell(:) = {'Anger'};
disgust_cell = cell(1,50);
disgust_cell(:) = {'Disgust'};
fear_cell = cell(1,50);
fear_cell(:) = {'Fear'};
happiness_cell = cell(1,50);
happiness_cell(:) = {'Happiness'};

emotions_labels = categorical([anger_cell disgust_cell fear_cell happiness_cell]);
imds = imageDatastore('datasets/images/images_to_use_experiment', 'labels', emotions_labels);

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
    
    maxPooling2dLayer(2,'Stride',2)   
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer  
    
    maxPooling2dLayer(2,'Stride',2)   
    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer  
    
    maxPooling2dLayer(2,'Stride',2)   
    
    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    reluLayer  
    
    maxPooling2dLayer(2,'Stride',2)   
    
    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    reluLayer  
    
    maxPooling2dLayer(2,'Stride',2)   
    
    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    reluLayer  
    
    
    fullyConnectedLayer(4)
    
    softmaxLayer
    
    classificationLayer
];


%% Training 
% The mini-batch size should be less than the data set size; the mini-batch 
% is used at each training iteration to evaluate gradients and update the 
% weights. 
options = trainingOptions('adam', ...
'InitialLearnRate', 0.05, ...
'MiniBatchSize', 20, ...
'MaxEpochs', 50, ...
'Shuffle','every-epoch', ...
'ValidationData', imdsTest, ...
'ValidationFrequency', 4, ...
'Verbose', false, ...
'Plots', 'training-progress');

net = trainNetwork(imdsTrain,layers,options);


%% 
predLabels = classify(net,imdsTest);
testLabels = imdsTest.Labels;
accuracy = sum(predLabels == testLabels)/numel(testLabels);
fprintf('Accuracy is %8.2f%%\n',accuracy*100)

% %% Implement and test the neural net to classify circles and ellipses
% % See also:
% % categorical, imageDatastore, countEachLabel, splitEachLabel, imageInputLayer,
% % convolution2dLayer, batchNormalizationLayer, reluLayer, maxPooling2dLayer,
% % softmaxLayer, classificationLayer, trainingOptions, trainNetwork, classify
% 
% 
% %% Get the images
% cd Ellipses_200
% type = load('Type');
% cd ..
% t    = categorical(type.t);
% imds = imageDatastore('Ellipses_200','labels',t);
% 
% labelCount = countEachLabel(imds);
% 
% % Display a few ellipses
% figure('Name', 'Ellipses');
% n = 4;
% m = 5;
% % random selection
% ks = sort(randi(length(type.t),1,n*m));
% for i = 1:n*m
% 	subplot(n,m,i);
% 	imshow(imds.Files{ks(i)});
%     title(sprintf('Image %d: %d',ks(i),type.t(ks(i))))
% end
% 
% % We need the size of the images for the input layer
% img = readimage(imds,1);
% 
% % Split the data into training and testing sets
% fracTrain = 0.8;
% [imdsTrain,imdsTest] = splitEachLabel(imds,fracTrain,'randomize');
% 
% 
% %% Define the layers for the net
% % This gives the structure of the convolutional neural net
% layers = [
%     imageInputLayer(size(img))  
%     
%     convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)   
%     
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer    
%     
%     maxPooling2dLayer(2,'Stride',2)   
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer  
%     
%     fullyConnectedLayer(2)
%     softmaxLayer
%     classificationLayer
% ];
% 
% 
% %% Training   
% % The mini-batch size should be less than the data set size; the mini-batch is
% % used at each training iteration to evaluate gradients and update the weights. 
% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.01, ...
%     'MiniBatchSize',16, ...
%     'MaxEpochs',5, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',imdsTest, ...
%     'ValidationFrequency',2, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
% 
%   
% net = trainNetwork(imdsTrain,layers,options);
% 
% 
% %% Test the neural net
% predLabels  = classify(net,imdsTest);
% testLabels  = imdsTest.Labels;
% 
% accuracy = sum(predLabels == testLabels)/numel(testLabels);
% fprintf('Accuracy is %8.2f%%\n',accuracy*100)