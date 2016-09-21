function hw1_p2(data, labels, testdata, testlabels, numsamples_list)
        N = size(data, 1);
        error_list = zeros(size(numsamples_list));
        error_stddev = zeros(size(error_list));
        for i = 1:size(numsamples_list, 2)
                n = numsamples_list(1, i)
                numruns = 1;
                error_run = zeros(1, numruns);
                for run = 1:numruns
                        [ndata, nlabels] = sel_prototype(data, labels, n);
                        preds = classify_1nn(ndata, nlabels, testdata, ...
                                                             testlabels);
                        error_run(1, run) = test_error_rate(preds, testlabels);
                end
                error_list(1, i) = sum(error_run)/numruns;
                error_stddev(1, i) = std(error_run);
        end
        save('p2_error_rates', 'error_list');
        figure;
        plot(numsamples_list, error_list);
        title('Learning Curve (Problem 2): Error % vs Size of Prototype');
        xlabel('Size of Prototype, m');
        ylabel('Error Rate in %');
end

function [ndata, nlabels] = sel_prototype(data, labels, n)
        [idx, ndata] = kmeans(data, n);
        data_idx = [1:size(data, 1)];
        nlabels = accumarray(idx, data_idx, [], @(x) mode(labels(x, :)));
end
