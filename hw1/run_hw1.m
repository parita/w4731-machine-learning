disp('Loading ocr.mat ...')
load('ocr.mat')
N = [1000, 2000, 4000, 8000];
disp('Running Problem 1 code ...')
hw1_p1(data, labels, testdata, testlabels, N);
disp('Running Problem 2 code ...')
hw1_p2(data, labels, testdata, testlabels, N);
