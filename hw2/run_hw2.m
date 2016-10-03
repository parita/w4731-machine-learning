% Load data
load('news.mat')
% Run hw2 Problem 1(b)
disp('Running Problem 1(b)...')
tic;
[training_error_rate, test_error_rate, ~, ~] = hw2_p1b(data, labels, testdata, testlabels)
toc;
% Run hw2 Problem 1(c) - creates a new dataset and calls hw2_p1b
disp('Running Problem 1(c)...')
tic;
[training_error_rate, test_error_rate] = hw2_p1c(data, labels, testdata, testlabels)
toc;
clear all;
% Run hw2 Problem 1(d) - Compute alpha0..j with the most positive and negative word
tic;
disp('Running Problem 1(d)...')
[alpha0, pos_words, neg_words] = hw2_p1d;
alpha0
pos_words'
neg_words'
toc;
