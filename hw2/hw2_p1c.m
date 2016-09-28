function [training_error_rate, test_error_rate] = hw2_p1c(data, labels, testdata, testlabels)
        [ndata, nlabels, ntestdata, ntestlabels] = create_binary_dataset(data, labels, testdata, testlabels);
        [training_error_rate, test_error_rate, pi_y, mu_y] = hw2_p1b(ndata, nlabels, ntestdata, ntestlabels);
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
