function error_rate = test_error_rate(preds, testlabels)
        error_rate = nnz(testlabels - preds)/size(testlabels, 1) * 100;
end
