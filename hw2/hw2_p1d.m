function [alpha0, pos, neg] = hw2_p1d(data, labels, testdata, testlabels)
        load('new_binary_news.mat');
        [training_error_rate, test_error_rate, pi_y, mu_y] = hw2_p1b(ndata, nlabels, ntestdata, ntestlabels);
        [alpha0, alpha] = get_alpha(pi_y, mu_y);
        [~, sortidx] = sort(alpha, 'descend');
        vocab = textread('news.vocab', '%s');
        pos = vocab(sortidx(1:20, :));
        neg = vocab(sortidx(end:-1:end-20+1, :));
end

function [alpha0, alpha] = get_alpha(p, m)
        log_1_minus_mu = sum(log(1 - m), 2);
        alpha0 = log(p(2, :)) - log(p(1, :)) + log_1_minus_mu(2, :) - log_1_minus_mu(1, :);
        alpha = log(m(2, :)) + log(1 - m(1, :)) - log(m(1, :)) - log(1 - m(2, :));
        alpha = alpha';
end
