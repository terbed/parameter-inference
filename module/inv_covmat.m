function [inv_covmat] = inv_covmat(D, lamb, t)
len = numel(t);
covmat = zeros(len,len);

for i = 1:len
    t1 = t(i);
    for j = 1:len
        t2 = t(j);
        covmat(i,j) = D * lamb * exp(-lamb * abs(t1 - t2));
    end
end

inv_covmat = inv(covmat);
