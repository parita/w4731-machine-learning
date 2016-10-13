function [data, labels] = hw3_prepocess()
	D = readtable('reviews_tr.csv', 'Delimiter', ',', 'Format', '%d%s');
	% data = regexp(D{1:200000, 2}, ' ', 'split');
	data = vertcat(D{1:200000, 2});
	labels = D{1:200000, 1};
	clear D;
end
