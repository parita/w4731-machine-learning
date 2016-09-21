function hw1_p1(data, labels, testdata, testlabels, numsamples_list)
        N = size(data, 1);
        error_for_n = zeros(size(numsamples_list));
        for i = 1:size(numsamples_list, 2)
                n = numsamples_list(1, i)
                for run = 1:10
                        sel = randsample(N, n);
                        ndata = data(sel, :);
                        nlabels = labels(sel, :);
                        preds = classify_1nn(ndata, nlabels, testdata, ...
                                                             testlabels);
                        error_for_n(1, i) = error_for_n(1, i) + ...
                                            test_error_rate(preds, testlabels);
                end
                error_for_n(1, i) = error_for_n(1, i)/10
        end
        figure;
        plot(numsamples_list, error_for_n);
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
        error_rate = nnz(testlabels - preds)/size(testlabels, 1);
end
