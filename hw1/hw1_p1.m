function hw1_p1(data, labels, testdata, testlabels, numsamples_list)
        N = size(data, 1);
        error_list = zeros(size(numsamples_list));
        error_stddev = zeros(size(error_list));
        for i = 1:size(numsamples_list, 2)
                n = numsamples_list(1, i)
                numruns = 10;
                error_run = zeros(1, numruns);
                for run = 1:numruns
                        sel = randsample(N, n);
                        ndata = data(sel, :);
                        nlabels = labels(sel, :);
                        preds = classify_1nn(ndata, nlabels, testdata, ...
                                                             testlabels);
                        error_run(1, run) = test_error_rate(preds, testlabels);
                end
                error_list(1, i) = sum(error_run)/numruns;
                error_stddev(1, i) = std(error_run);
        end
        figure;
        errorbar(numsamples_list, error_list, error_stddev);
        title('Learning Curve (Problem 1): Error % vs Size of Training Data');
        xlabel('Size of Training Data, n');
        ylabel('Error Rate in %');
end

function preds = classify_1nn(data, labels, testdata, testlabels)
        preds = zeros(size(testdata, 1), 1);
        x2 = sum(testdata.^2, 2);
        x2 = repmat(x2', size(data, 1), 1);
        xy = data*testdata';
        y2 = sum(data.^2, 2);
        y2 = repmat(y2, 1, size(testdata, 1));
        dist = x2 + y2 - 2*xy;
        clear xy y2 x2;
        [~, idx] = min(dist);
        i = [1:size(testdata, 1)];
        preds(i, :) = labels(idx, :);
end

function error_rate = test_error_rate(preds, testlabels)
        error_rate = nnz(testlabels - preds)/size(testlabels, 1) * 100;
end
