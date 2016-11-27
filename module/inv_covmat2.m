function [inv_covmat2] = inv_covmat2(D, lamb, A, mu, sig, t)
len = numel(t);
covmat = zeros(len,len);

for i = 1:len
    t1 = t(i);
    for j = 1:len
        t2 = t(j);
        covmat(i,j) = D*lamb*exp(-lamb*abs(t1-t2)) - A*exp(-(abs(t1-t2)-mu)^2/(2*sig^2));
    end
end

inv_covmat2 = inv(covmat);