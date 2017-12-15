function [covmt] = covmat(D, lamb, t)
len = numel(t);
covmt = zeros(len,len);

for i = 1:len
    t1 = t(i);
    for j = 1:len
        t2 = t(j);
        covmt(i,j) = D * lamb * exp(-lamb * abs(t1 - t2));
    end
end