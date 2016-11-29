function mae = hw5_p2b(data, labels, testdata, testlabels, n, tol)
	% Lift data to make the problem homogenous
	data = [data, ones(size(data, 1), 1)];
    testdata = [testdata, ones(size(testdata, 1), 1)];
	beta = zeros(size(data, 2), 1);
	init_step_size = 1;
	beta = gradient_descent(data, labels, beta, n, tol, init_step_size);
    n = size(testdata, 1);
    k = 128;
    mae = MAE(P(testdata, testlabels, beta, n/k), Pi(testlabels, n, k), n/k)
end

function beta = gradient_descent(x, y, b, n, tol, eta)
	iter = 0;
	while f(x, y, b, n) > tol
		del = compute_gradient(x, y, b, n);
		eta1 = line_search(x, y, b, del, n, 1);
		b = b - eta1.*del;
		iter = iter + 1;
	end
	final_obj = f(x, y, b, n)
	iterations = iter
	beta = b
end

function del = compute_gradient(x, y, b, n)
	bx = x*b;
	L = (1./(1 + exp(-bx))) - y;
	L = kron(ones(1, size(x, 2)), L);
	del = 1./n.*sum(L.*x)';
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

function cp = P(x, y, b, n)
    x = x(1:n, :);
    bx = x*b;
    cp = 1./(1 + exp(-bx));
end

function pi = Pi(y, n, k)
    subs = [1:n/k]';
    subs = kron(ones(k, 1), subs);
    pi = 1/k*accumarray(subs, y);
end

function error = MAE(P, pi, n)
    error = 1/n*sum(abs(P - pi));
end