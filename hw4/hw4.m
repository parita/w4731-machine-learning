function hw4()
	load('hw4data')
	n = size(data, 1);
	disp('Running Problem 2(b) ...')
	hw4_p2b(data, labels, n)

	disp('Transforming data for Problem 2(c) ...')
	t_data = transform_data(data);
	disp('Running Problem 2(c) ...')

	hw4_p2b(t_data, labels, n)

	disp('Running Problem 2(d) ...')
	disp('Part 1: Run modified gradient descent with original data')
	hw4_p2d(data, labels, n)

	disp('Transforming data for Problem 2(d) Part 2 ...')
	t_data = transform_data(data);
	disp('Part 2: Run modified gradient descent with transformed data')
	hw4_p2d(t_data, labels, n)
end

function x_T = transform_data(x)
	x1_max = max(x(:, 1));
	x2_max = max(x(:, 2));
	x3_max = max(x(:, 3));
	A = [1/x1_max, 0, 0; 0, 1/x2_max, 0; 0, 0, 1/x3_max]
	x_T = x*A;
end
