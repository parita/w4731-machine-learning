function hw5()
	load('hw5data')
	n = size(data, 1);
	disp('Running Problem 2(b) ...')
	error = hw5_p2b(data, labels, n, 0.50317)
    clear all;
    disp('Running Problem 3(b.1) ...')
    load('Housing');
    n = size(data, 1);
    disp('OLS')
    error = ols(data, labels, testdata, testlabels, n)
    disp('CoSaMP')
    error = sparselsq(data, labels, testdata, testlabels, n, 3)
end

function error = ols(data, labels, testdata, testlabels, n)
    [w, ~] = mvregress(data, labels);
    E = testlabels - testdata*w;
    error = sum(E.^2)/n;
end

function error = sparselsq(data, labels, testdata, testlabels, n, k)
    features = {'Threshold', 'CRIM', 'ZN', 'INDUS', 'CHAS', ...
                'NOX', 'RM', 'AGE', 'DIS', 'RAD', ...
                'TAX', 'PTRATIO', 'B', 'LSTAT'};
    w = OMP(data, labels, k + 1);
    idx = find(w ~= 0);
    features(idx)
    E = testlabels - testdata*w;
    error = sum(E.^2)/n;
end