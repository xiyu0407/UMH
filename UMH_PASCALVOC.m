%% @author Jun Yu
% Jiangnan University
% function [MAP_I2T,MAP_T2I] = PASCALUH1TO2(beta)
clear;
clc;
datastr='PascalVOC';
iskernel = 1;
rng('default');
rng(0) ;
%% dataset
if strcmp(datastr,'PascalVOC')
    load('./Data/Pascal_voc.mat');
end
X_Train = I_tr(sampleInds,:); Y_Train =  T_tr(sampleInds,:); 
MEANX = mean(X_Train,1); MEANY = mean(Y_Train,1); 
X_Train1 =  X_Train - MEANX;  Y_Train1 =  Y_Train - MEANY;
clear X_Train Y_Train
X_Test1 =  I_te - MEANX;  Y_Test1 =  T_te - MEANY;  
X_Retrieval1 =  I_tr - MEANX;  Y_Retrieval1 =  T_tr - MEANY;
if size(L_tr,2) == 1 || size(L_tr,1) == 1
    L_tr = sparse(1:length(L_tr), double(L_tr), 1); L_tr =full(L_tr);
    L_te = sparse(1:length(L_te), double(L_te), 1); L_te =full(L_te); 
end 
LTrain = L_tr(sampleInds,:);  LTest = L_te; LRetrieval = L_tr;
clear I_tr T_tr I_te T_te L_tr L_te;
% L = LTrain';
%% initialization
bits = [64];% the length of bits
if strcmp(datastr,'PascalVOC')
    rho = 1e-3;%j 1e-3
    lamda2 = 10;% 1e-3
    lamda1 = 1e-4;
    knum = 2500;
    beta = 0.1;
end
T_iter = 80;% the number of iteration in the Algorithm%%
Totaltimes = 1;
gamma = 0.5;
statvalue_NLM_Non = zeros(Totaltimes,2,numel(bits));
%%
for runtime=1 :Totaltimes
    %  RBF Kernel
    if iskernel==1
        sampleType = 'random';%'random' or 'Kmeans'
        kernelfunctiontype =0;% kernel function
        [X_Train,Y_Train,X_Test,Y_Test,X_Retrieval,Y_Retrieval] = GET_Kernel_matrix(X_Train1,Y_Train1,X_Test1,Y_Test1,X_Retrieval1,Y_Retrieval1,kernelfunctiontype,knum,sampleType);
    else
        X_Train = X_Train1'; Y_Train = Y_Train1';X_Test = X_Test1';Y_Test = Y_Test1'; X_Retrieval = X_Retrieval1'; Y_Retrieval = Y_Retrieval1';
    end
    Y1 = (Y_Train'./ repmat(sqrt(sum(Y_Train'.*Y_Train',2)),[1 size(Y_Train',2)]))';
    clear X_Train1 Y_Train1 X_Test1 Y_Test1 X_Retrieval1 Y_Retrieval1
    N = size(X_Train,2);% N is the number of Image in training set
    X =X_Train;Y = Y_Train; 
    if exist(sprintf('./Data/%sS.mat',datastr),'file')
         load(sprintf('./Data/%sS',datastr));
    else
        S = getSimilarMatrix(X,ceil(0.1 * size(X,2)));
        save(sprintf('./Data/%sS.mat',datastr),'S')
    end
    %% Iteration procedure
    history=struct('iter','objval');
    fprintf('-------start------\n');
    for j=1:numel(bits)
        r = bits(j); 
        P = randn(size(X,1),r);
        Q = randn(size(Y,1),r);
        B = sign(randn(N,r));
        alpha1=0.5;alpha2=0.5;
        for iter=1:T_iter
            % optimization of B  
            R = alpha1 ^ gamma * X' * P + alpha2 ^ gamma * Y' * Q;
            B = sign(((-2 * S + S' * S + rho / 2 * ones(N,1) * ones(N,1)' - beta/4 * (Y1') * Y1 + 0.01 * trace(-2 * S + S' * S )* eye(N))) \ R);
            % optimization of D
             itervalue_d = zeros(size(P,1),1);
             for iterd = 1 : size(P,1) 
                 itervalue_d(iterd) = 1 / (2 * sqrt(P(iterd,:) * P(iterd,:)')+ 1e-5);
             end
             Dd = diag(itervalue_d);
             % optimization of Z
             itervalue_z = zeros(size(Q,1),1);
             for iterz = 1 : size(Q,1) 
                 itervalue_z(iterz) = 1 / (2 * sqrt(Q(iterz,:) * Q(iterz,:)')+ 1e-5);
             end
             Dz = diag(itervalue_z);
             % optimization of P
             P = (X * X' + lamda1 * Dd) \ X * B;
             % optimization of Q
             Q = (Y * Y' + lamda2 * Dz) \ Y * B;
            % optimization of alpha
             Gx = norm(X' * P - B,'fro')^2 + lamda1 * getL21norm(P);
             Gy = norm(Y' * Q  - B,'fro')^ 2 + lamda2 * getL21norm(Q);
             alpha1 = (Gx ^ (1 / (1 - gamma)) ) / (Gx ^ (1 / (1 - gamma) ) + Gy ^ (1 / (1 - gamma)));
             alpha2 = (Gy ^ (1 / (1 - gamma)) ) / (Gx ^ (1 / (1 - gamma) ) + Gy ^ (1 / (1 - gamma)));   

             value1 = alpha1 ^ gamma * (norm(X' * P - B,'fro')^2 + lamda1 * getL21norm(P));
             value2 = alpha2 ^ gamma * (norm(Y' * Q - B,'fro')^2 + lamda2 * getL21norm(Q));     
             value3 =  norm(B - S * B,'fro') ^ 2;
             value4 =  rho * norm(ones(1,N) * B,'fro') ^ 2;
             value5 = -beta/2 * norm(Y1 * B, 'fro') ^ 2;
             value =  value1 + value2 + value3 + value4 + value5; 
            history.iter(iter)=iter;
            history.objval(iter)=value;
           fprintf('第%s次迭代: 第一项输出值：%s; 第二项输出值：%s;第三项输出值：%s;第四项输出值：%s;第五项输出值：%s;总的输出值为：%s\n',...
               num2str(iter),num2str(value1), num2str(value2), num2str(value3),num2str(value4), num2str(value5),num2str(value));
        end 
        save('./Data/model.mat','P','Q','B','alpha1','alpha2');
      
      %% Convergence curve
      if r == 64
            figure('Color',[1 1 1]);
%             subplot(1,2,1);
            Dimension = 1:1:iter;
            plot(Dimension,history.objval,'-r<','LineWidth',1,...
                                    'MarkerEdgeColor','r',...
                                    'MarkerFaceColor','r',...
                                    'MarkerSize',4);
            ylabel('Step1: objective function value');
            xlabel('iteration number');
      end
        %% evaluation    
        I_tr=defsign(X_Retrieval' * P); 
        I_te=defsign(X_Test' * P);
        T_tr=defsign(Y_Retrieval' * Q);
        T_te=defsign(Y_Test' * Q);

        I_tr=compactbit((I_tr>=0));
        I_te=compactbit((I_te>=0));
        T_tr=compactbit((T_tr>=0));
        T_te=compactbit((T_te>=0)); 
        hammingM = hammingDist(I_te, T_tr)';
        MAP_I2T= perf_metric4Label( LRetrieval,LTest, hammingM );   
        fprintf('I2T_MAP值为：%s\n',MAP_I2T);
        hammingM = hammingDist(T_te, I_tr)';
        MAP_T2I = perf_metric4Label( LRetrieval,LTest, hammingM );
        fprintf('T2I_MAP值为：%s\n',MAP_T2I);
        statvalue_NLM_Non(runtime,1,j) = MAP_I2T;
        statvalue_NLM_Non(runtime,2,j) = MAP_T2I;
    end
end
