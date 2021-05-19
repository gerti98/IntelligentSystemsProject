%% Preparation of data
clear
close all
clc

%Load datasets
dataset = load('C:\Users\gxhan\Documents\Universita\Esami_Correnti\Intelligent Systems\IntelligentSystemsProject\datasets\dataset.mat'); 
dataset = table2array(dataset.dataset);