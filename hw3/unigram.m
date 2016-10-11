function [ndata, tf, idf] = unigram(data)
	for i = 1:size(data, 1)
		if (mod(i, 10000) == 0)
			Elements_processed = i
		end
		words = strsplit(data{i, :});
		[a, ~, ia] = unique(words);
		ndata{i, 1} = a;
		tf{i, 1} = accumarray(ia, 1);
	end
	nD = 200000;
	all_words = unique(strsplit(strjoin(data)));
	idf = zeros(size(all_words));
	for i = 1:size(all_words, 2)
		tindex = strfind(data, all_words(1, i));
		index = find(not(cellfun('isempty', tindex)));
		idf(1, 1) = nD/size(index, 1);
	end
end
