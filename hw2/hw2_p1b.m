function [training_error_rate, test_error_rate, pi_y, mu_y] = ...
                                hw2_p1b(data, labels, testdata, testlabels)
        [pi_y, mu_y] = get_laplace_params(data, labels);
        preds = classify(data, mu_y, pi_y);
        training_error_rate = get_error_rate(preds, labels);
        preds = classify(testdata, mu_y, pi_y);
        test_error_rate = get_error_rate(preds, testlabels);
        clear preds
end

function [pi_y, mu_y] = get_params(data, labels)
        count_y = accumarray(labels, 1);
        pi_y = count_y./size(labels, 1);
        data_idx = [1: size(data, 1)];
        labels_idx = [1:size(count_y, 1)];
        mu_y = accumarray(labels, data_idx, [], @(x) {sum(data(x, :))./size(x, 1)});
        mu_y(cellfun(@isempty, mu_y)) = {zeros(1, size(mu_y, 2))};
        mu_y = cell2mat(mu_y);
        clear count_y data_idx labels_idx
end

function [pi_y, mu_y] = get_laplace_params(data, labels)
        count_y = accumarray(labels, 1);
        pi_y = count_y./size(labels, 1);
        data_idx = [1: size(data, 1)];
        labels_idx = [1:size(count_y, 1)];
        mu_y = accumarray(labels, data_idx, [], @(x) {(1 + sum(data(x, :)))./(2 + size(x, 1)) });
        mu_y(cellfun(@isempty, mu_y)) = {zeros(1, size(mu_y, 2))};
        mu_y = cell2mat(mu_y);
        clear count_y data_idx labels_idx
end

function preds = classify(testdata, mu_y, pi_y)
        log_pi_y = repmat(log(pi_y), 1, size(testdata, 1));
        % g = log_pi_y + log(mu_y)*testdata' + log(1 - mu_y)*(1 - testdata)';
        [~, preds] = max(log_pi_y + log(mu_y)*testdata' + log(1 - mu_y)*(1 - testdata)');
        clear log_pi_y g;
        preds = preds';
end

function error_rate = get_error_rate(preds, testlabels)
        error_rate = nnz(testlabels - preds)/size(testlabels, 1) * 100;
end
