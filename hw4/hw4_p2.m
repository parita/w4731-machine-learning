function hw4_p2()
	load('hw4data')
	n = size(data, 1);
	hw4_p2b(data, labels, n)
	hw4_p2c(data, labels, n)
	% hw4_p2s(data, labels, n)
end

function hw4_p2d(data, labels, n)
	% Segregate data into training and hold-out set
	ndata_size = int32(n*0.8)
	ntestdata_size = n - ndata_size
	ndata = data(1:ndata_size, :)
	nlabels = labels(1:ndata_size, :)
	ntestdata = data(ndata_size + 1: end)
	ntestlabels = labels(ndata_size + 1: end)

	% Apply modified gradient descent
	beta = zeros(size(data, 2), 1);
	gradient_descent_mod(data, labels, beta, n, 1);
end

function hw4_p2b(data, labels, n)
	% Lift data to make the problem homogenous
	data = [data, ones(size(data, 1), 1)];
	beta = zeros(size(data, 2), 1);
	gradient_descent(data, labels, beta, n, 0.65064, 1);
end

function hw4_p2c(data, labels, n)
	plot_data(data, labels);
	% Scale data for each feature
	data = transform_data(data);
	plot_data(data, labels);
	% Lift data to make the problem homogenous
	data = [data, ones(size(data, 1), 1)];
	beta = zeros(size(data, 2), 1);
	gradient_descent(data, labels, beta, n, 0.65064, 1);
end

function gradient_descent(x, y, b, n, tol, eta)
	iter = 0;
	while f(x, y, b, n) > tol
		bx = x*b;
		L = (1./(1 + exp(-bx))) - y;
		L = kron(ones(1, size(x, 2)), L);
		del = 1./n.*sum(L.*x)';
		eta1 = line_search(x, y, b, del, n, 1);
		b = b - eta1.*del;
		if (mod(iter, 100) == 0)
			obj = f(x, y, b, n)
		end
		iter = iter + 1;
	end
	iterations = iter
	beta = b
end

function eta = line_search(x, y, b, del, n, eta)
	fb_del = f(x, y, b - eta.*del, n);
	fb = f(x, y, b, n);
	iter = 0;
	while fb_del > (fb - 0.5*eta*(norm(del)^2))
		eta = eta/2;
		fb_del = f(x, y, b - eta.*del, n);
		fb = f(x, y, b, n);
		iter = iter + 1;
	end
end

function L = f(x, y, b, n)
	bx = x*b;
	L = log(1 + exp(bx)) - y.*bx;
	L = 1/n*sum(L);
end

function plot_data(x, y)
	x1 = x(find(y == 0), :);
	x2 = x(find(y == 1), :);
%	figure;
%	h = scatter3(x(:, 1), ones(size(x(:, 1))), ones(size(x(:, 1))), 36, 'r');
%	hold on;
%	h = scatter3(ones(size(x(:, 1))), x(:, 2), ones(size(x(:, 1))), 36, 'b');
%	hold off;
%	hold on;
%	h = scatter3(ones(size(x(:, 1))), ones(size(x(:, 1))), x(:, 3), 36, 'g');
%	hold off;
%	h.MarkerFaceColor = [0 0.5 0.5];
	figure;
	h = scatter3(x1(:, 1), x1(:, 2), x1(:, 3), 36, 'r');
	hold on;
	h = scatter3(x2(:, 1), x2(:, 2), x2(:, 3), 36, 'r');
	hold off;
	h.MarkerFaceColor = [0 0.5 0.5];
end

function x_T = transform_data(x)
	x1_max = max(x(:, 1));
	x2_max = max(x(:, 2));
	x3_max = max(x(:, 3));
	A = [1/x1_max, 0, 0; 0, 1/x2_max, 0; 0, 0, 1/x3_max];
	x_T = x*A;
end
