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
