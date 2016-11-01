function hw4_p2d(data, labels, n)
	% Lift data to make the problem homogenous
	xdata = [data, ones(size(data, 1), 1)];

	% Segregate data into training and hold-out set
	[ndata, nlabels, ntestdata, ntestlabels, ndata_size, ntestdata_size] = ...
		split_data(xdata, labels, n, 0.8);

	% Apply modified gradient descent
	beta = zeros(size(ndata, 2), 1);
	init_step_size = 1;
	err_tol = 0.99;
	gradient_descent_mod(ndata, nlabels, ntestdata, ntestlabels, ...
				beta, ndata_size, ntestdata_size, ...
				err_tol, init_step_size);

end

function [ndata, nlabels, ntestdata, ntestlabels, ndata_size, ntestdata_size] = ...
	split_data(data, labels, n, fraction)
	% Segregate data into training and hold-out set
	ndata_size = floor(n*fraction);
	ntestdata_size = n - ndata_size;
	ndata = data(1:ndata_size, :);
	nlabels = labels(1:ndata_size, :);
	ntestdata = data(ndata_size + 1: end, :);
	ntestlabels = labels(ndata_size + 1: end, :);
end

function gradient_descent_mod(x, y, test_x, test_y, b, n_tr, n_te, err_tol, eta)
	iter = 1;
	err_rate = 0;
	best_err_rate = inf;
	while 1
		bx = x*b;
		L = (1./(1 + exp(-bx))) - y;
		L = kron(ones(1, size(x, 2)), L);
		del = 1./n_tr.*sum(L.*x)';
		eta1 = line_search(x, y, b, del, n_tr, 1);
		b = b - eta1.*del;
		if (floor(log2(iter)) == log2(iter))
			obj = f(x, y, b, n_tr);
			err_rate = compute_error(test_x, test_y, n_te, b);
			if(iter >= 32 && err_rate > err_tol*best_err_rate )
				break;
			end
			if err_rate < best_err_rate
				best_err_rate = err_rate;
			end
		end
		iter = iter + 1;
	end
	beta = b
	final_obj = f(x, y, b, n_tr)
	iterations = iter
	err_rate
end

function err_rate = compute_error(test_x, test_y, n_te, b)
	pred = zeros(size(test_y));
	bx = logistic(test_x*b);
	pred(bx <= 0.5) = 0;
	pred(bx > 0.5) = 1;
	err_rate = sum(pred ~= test_y)./n_te*100;
end

function val = logistic(z)
	val = 1./(1 + exp(-z));
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
