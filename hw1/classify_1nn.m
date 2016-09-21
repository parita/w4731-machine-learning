function preds = classify_1nn(data, labels, testdata, testlabels)
        preds = zeros(size(testdata, 1), 1);
        xy = data*testdata';
        [~, idx] = max(xy);
        i = [1:size(testdata, 1)];
        preds(i, :) = labels(idx, :);
end

function error_rate = test_error_rate(preds, testlabels)
        error_rate = nnz(testlabels - preds)/size(testlabels, 1)
end

function main(data, labels, testdata, testlabels, numsamples_list)
        N = size(data, 1);
        error_for_n = zeros(1, size(numsamples_list, 2));
        for i = 1:size(numsamples_list, 2)
                n = numsamples_list[i]
                for run = 1:10
                        sel = randsample(N, n);
                        ndata = data(sel, :);
                        nlabels = labels(sel, :);
                        preds = classify_1nn(ndata, nlabels, testdata, ...
                                                             testlabels);
                        error_for_n(1, n) = error_for_n(1, n) + ...
                                            test_error_rate(preds, testlabels);

                end
end
