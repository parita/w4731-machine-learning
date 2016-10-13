function [ndata, tf] = bigram(data)
	for i = 1:size(data, 1)
		if(mod(i, 10000) == 0)
			Elements_processed = i
		end
		biwords = regexp(data{i, 1}, '\w+ \w+', 'match');
		[a, ~, ia] = unique(biwords);
		ndata{i, 1} = a;
		tf{i, 1} = accumarray(ia, 1);
end
