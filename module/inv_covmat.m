function [ inv_covmat] = inv_covmat(D, lamb, t)
len = numel(t);
covmat = zeros(len,len);

for i = 1:len
    t1 = t(i);
    for j = 1:len
        t2 = t(j);
        covmat(i,j) = D * lamb * exp(-lamb * abs(t1 - t2));
    end
end

S = sparse(covmat);
C = inv(S);
inv_covmat = full(C);

% Run function with these settings:
% t = linspace(0,1500,30000)
% invcovmat = inv_covmat(30, 0.1, t)
