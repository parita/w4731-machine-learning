function [ndata, tf, tf_idf] = unigram(data)
	nD = 200000;
	disp('Computing tf and tf-idf for unigrams ...')
	for i = 1:10 % size(data, 1)
		if (mod(i, 1000) == 0)
			Elements_processed = i
		end
		words = strsplit(data{i, :});
		[a, ~, ia] = unique(words);
		ndata{i, 1} = a;
		tmp = accumarray(ia, 1);
		tmp = tmp';
		tf{i, :} = tmp;
		for j = 1:size(a, 2)
			tf_idf{i, j} = tmp(:, j)*log10(wordidf(char(a(:, j)), data, nD));
		end
	end
end

function idf = wordidf(word, data, nD)
	tindex = strfind(data, word);
	index = find(not(cellfun('isempty', tindex)));
	idf = nD/size(index, 1);
end
