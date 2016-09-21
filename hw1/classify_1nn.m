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
