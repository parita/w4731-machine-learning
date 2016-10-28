function hw4_p2()
	load('hw4data')
	% Lift data to make the problem homogenous
	data = [data, ones(size(data, 1), 1)];
	% labels(find(labels == 0)) = -1;
	n = size(data, 2);
	beta = zeros(size(data, 2), 1);
	gradient_descent(data, labels, beta, n, 0.65064, 1);
end

function gradient_descent(x, y, b, n, tol, eta)
	size(b)
	size(x)
	iter = 0;
	while f(x, y, b, n) > tol
		bx = x*b;
		L = (1./(1 + exp(-bx))) - y;
		L = kron(ones(1, size(x, 2)), L);
		del = 1./n.*sum(L.*x)';
		eta1 = line_search(x, y, b, del, n, 1);
		% eta1 = eta1 / 2;
		b = b - eta1.*del;
		obj = f(x, y, b, n)
		iter = iter + 1;
	end
	iter
end

function eta = line_search(x, y, b, del, n, eta)
	fb_del = f(x, y, b - eta.*del, n);
	fb = f(x, y, b, n);
	while fb_del > (fb - 0.5*eta*(norm(del)^2))
		fb_del = f(x, y, b - eta.*del, n);
		fb = f(x, y, b, n);
		eta = eta/2;
	end
	eta = eta*2;
end
function L = f(x, y, b, n)
	bx = x*b;
	L = log(1 + exp(bx)) - y.*bx;
	L = 1/n*sum(L);
end
