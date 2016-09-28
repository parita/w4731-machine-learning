function [preds, error_rate] = hw2_p1c(data, labels, testdata, testlabels)
        [ndata, nlabels, ntestdata, ntestlabels] = create_binary_dataset(data, labels, testdata, testlabels);
        [preds, error_rate, pi_y, mu_y] = hw2_p1b(ndata, nlabels, ntestdata, ntestlabels);
        [alpha0, alpha] = get_alpha(pi_y, mu_y);
        [~, sortidx] = sort(alpha, 'descend');
        vocab = textread('news.vocab', '%s');
        pos = vocab(sortidx(1:20, :));
        neg = vocab(sortidx(end-20+1:end, :));
end

function [ndata, nlabels, ntestdata, ntestlabels] = create_binary_dataset(data, labels, testdata, testlabels)
        [rclass0, ~] = find(ismember(labels, [1, 16, 20]));
        [rclass1, ~] = find(ismember(labels, [17, 18, 19]));
        ndata = data([rclass0; rclass1], :);
        nlabels = [ones(size(rclass0)); ones(size(rclass1)).*2];
        [rclass0, ~] = find(ismember(testlabels, [1, 16, 20]));
        [rclass1, ~] = find(ismember(testlabels, [17, 18, 19]));
        ntestdata = testdata([rclass0; rclass1], :);
        ntestlabels = [ones(size(rclass0)); ones(size(rclass1)).*2];
        save('new_binary_news.mat', 'ndata', 'nlabels', 'ntestdata', 'ntestlabels');
end

function [alpha0, alpha] = get_alpha(p, m)
        log_1_minus_mu = sum(log(1 - m), 2);
        alpha0 = log(p(2, :)) - log(p(1, :)) + log_1_minus_mu(2, :) - log_1_minus_mu(1, :);
        alpha = log(m(2, :)) + log(1 - m(1, :)) - log(m(1, :)) - log(1 - m(2, :));
        alpha = alpha';
end
