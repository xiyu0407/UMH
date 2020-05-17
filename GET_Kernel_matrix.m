% X is one modality, Y is another modality, Z is kernel base  
% Each ROW represents a sample
function [KX_Train,KY_Train,KX_Test,KY_Test,KX_Retrieval,KY_Retrieval] = GET_Kernel_matrix(X_Train,Y_Train,X_Test,Y_Test,X_Retrieval,Y_Retrieval,kernelfunctiontype,kernelSampleNum,sampleType)
%% RBF Kernel
    trainNum = size(X_Train,1);
%      kernelSampleNum = ceil(kernelanchorrate * trainNum);
%     kernelSampleNum = ceil(0.3 * trainNum);
%     kernelSampleNum = 1000;
    z = X_Train * X_Train';
    z = repmat(diag(z), 1, trainNum)  + repmat(diag(z)',trainNum, 1) - 2 * z;
    k1 = {};
    k1.type = kernelfunctiontype;
    k1.param = mean(z(:));                                  %  $\sigma^2$ for RBF kernel in image view

    z = Y_Train * Y_Train';
    z = repmat(diag(z), 1, trainNum)  + repmat(diag(z)', trainNum, 1) - 2 * z;
    k2 = {};
    k2.type = kernelfunctiontype;
    k2.param = mean(z(:));
    if strcmp(sampleType,'random')
        kernelSamples = sort(randperm(trainNum, kernelSampleNum));
        kernelX = X_Train(kernelSamples, :);
        kernelY = Y_Train(kernelSamples, :);
    elseif strcmp(sampleType,'Kmeans')
        opts = statset('Display', 'off', 'MaxIter', 100);
        [INX, C] = kmeans(X_Train, kernelSampleNum, 'Start', 'sample', 'EmptyAction', 'singleton', 'Options', opts, 'OnlinePhase', 'off');
         kernelX = C;
        [INX, C] = kmeans(Y_Train, kernelSampleNum, 'Start', 'sample', 'EmptyAction', 'singleton', 'Options', opts, 'OnlinePhase', 'off');
        kernelY = C;
    end
    KX_Train = kernelMatrix( kernelX, X_Train, k1 );
    KY_Train = kernelMatrix( kernelY, Y_Train, k2 );
    KX_Test = kernelMatrix( kernelX, X_Test, k1 );
    KY_Test = kernelMatrix( kernelY, Y_Test, k2 );
    if (size(X_Retrieval) ~= 0)
        KX_Retrieval = kernelMatrix( kernelX, X_Retrieval, k1 );
        KY_Retrieval = kernelMatrix( kernelY, Y_Retrieval, k2 );
    else
        KX_Retrieval = [];
        KY_Retrieval = [];
    end
end
