function hw1_p2(data, labels, testdata, testlabels, numsamples_list)
        N = size(data, 1);
        error_list = zeros(size(numsamples_list));
        error_stddev = zeros(size(error_list));
        for i = 1:size(numsamples_list, 2)
                n = numsamples_list(1, i)
                [ndata, nlabels] = sel_prototype(data, labels, n);
                preds = classify_1nn(ndata, nlabels, testdata, ...
                                                     testlabels);
                error_list(1, i) = test_error_rate(preds, testlabels)
                clear ndata nlabels preds
        end

        save('p2_error_rates', 'error_list');
        figure;
        plot(numsamples_list, error_list);
        title('Learning Curve (Problem 2): Error % vs Size of Prototype');
        xlabel('Size of Prototype, m');
        ylabel('Error Rate in %');
end

function [ndata, nlabels] = sel_prototype(data, labels, n)
        ndata = [];
        nlabels = [];
        for i = 0:9
                [r, c] = find(labels == i);
                d = data(r, :);
                l = labels(r, :);
                [idx, subset_ndata] =  fkmeans(d, n/10);
                data_idx = [1:size(d, 1)];
                subset_nlabels = accumarray(idx, data_idx, [], @(x) mode(l(x, :)));
                ndata = [ndata; subset_ndata];
                nlabels = [nlabels; subset_nlabels];
        end
end
