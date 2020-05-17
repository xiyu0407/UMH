function [ kM ] = kernelMatrix( trXA, trXB, kernelObj )
% calculation of a kernel matrix
% trXA: n1*f feature matrix, n1: the number of instances, f: feature dimensionality
% trXB: n2*f feature matrix, n2: the number of instances, f: feature dimensionality
% kernelObj: kernel settings, RBF kernel here
kernelType = kernelObj.type;
trmA = size(trXA, 1);
trmB = size(trXB, 1);
%RBF
if kernelType == 0
    parameter = kernelObj.param;
    tmp0 = trXA * trXB';
    dA = sum(trXA .* trXA, 2);
    dB = sum(trXB .* trXB, 2);
    Cov = repmat(dA, 1, trmB) + repmat(dB', trmA, 1) - 2 * tmp0;
    kM = exp(-(Cov ./ parameter(1)));
% 多项式
elseif kernelType == 1
    temp = trXA * trXB';
    c = ones(trmA,trmB);
    kM = (temp + c).^ 5;
%逆多元二次核（Inverse Multiquadric Kernel）
elseif kernelType == 2
    tmp0 = trXA * trXB';
    dA = sum(trXA .* trXA, 2);
    dB = sum(trXB .* trXB, 2);
    Cov = repmat(dA, 1, trmB) + repmat(dB', trmA, 1) - 2 * tmp0;
    kM = sqrt(Cov + ones(trmA,trmB)) .^ -1;
%多元二次核（Multiquadric Kernel）
elseif kernelType == 3
    tmp0 = trXA * trXB';
    dA = sum(trXA .* trXA, 2);
    dB = sum(trXB .* trXB, 2);
    Cov = repmat(dA, 1, trmB) + repmat(dB', trmA, 1) - 2 * tmp0;
    kM = sqrt(Cov + ones(trmA,trmB));
 % sigmoid
elseif kernelType == 4
    temp =0.01 * trXA * trXB';
    c = ones(trmA,trmB);
    kM = tanh(temp + c);
% k(x, y) = exp(-gamma Sum [(x - y)^2 / (x + y)])
elseif kernelType == 5
end
end

