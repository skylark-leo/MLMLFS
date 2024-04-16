function model = LSML( X, Y, optmParameter)
% This function is designed to Learn label-Specific Features and Class-Dependent Labels for Multi-Label Classification
% 
%    Syntax
%
%       [model] = LSML( X, Y, optmParameter)
%
%    Input
%       X               - a n by d data matrix, n is the number of instances and d is the number of features 
%       Y               - a n by l label matrix, n is the number of instances and l is the number of labels
%       optmParameter   - the optimization parameters for LSML, a struct variable with several fields, 
%
%    Output
%
%       model    -  a structure variable composed of the model coefficients

   %% optimization parameters
    lambda1          = optmParameter.lambda1; % missing labels
    lambda2          = optmParameter.lambda2; % regularization of W
    lambda3          = optmParameter.lambda3; % regularization of C
    lambda4          = optmParameter.lambda4; % regularization of graph laplacian
    lambda5          = optmParameter.lambda5; % regularization of M
    rho              = optmParameter.rho;
    eta              = optmParameter.eta;
    isBacktracking   = optmParameter.isBacktracking;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;

    num_dim   = size(X,2);
    num_class = size(Y,2);
    XTX = X'*X;
    XTY = X'*Y;
    YTY = Y'*Y;
    
   %% initialization
    W   = (XTX + rho*eye(num_dim)) \ (XTY); 
    W_1 = W; W_k = W;
    C = zeros(num_class,num_class); %eye(num_class,num_class);
    C_1 = C;
    Z   = sparse(num_dim,num_class);
    Z_1 = Z;
    iter = 1; oldloss = 0;
    bk = 1; bk_1 = 1; 
    Lip1 = 2*norm(XTX)^2 + 2*norm(-XTY)^2 + 2*norm((lambda1+1)*YTY)^2;
    Lip = sqrt(Lip1);
    while iter <= maxIter
       L = diag(sum(C,2)) - C;
       if isBacktracking == 0
           if lambda4>0
               Lip2 = norm(lambda4*(L+L'));
               Lip = sqrt( Lip1 + 2*Lip2^2);
           end
       else
           F_v = calculateF(W, XTX, XTY, YTY, C, lambda1, lambda4,Z);
           QL_v = calculateQ(W, XTX, XTY, YTY, C, lambda1, lambda4, Lip,W_k,Z);
           while F_v > QL_v
               Lip = eta*Lip;
               QL_v = calculateQ(W, XTX, XTY, YTY, C, lambda1, lambda4, Lip,W_k,Z);
           end
       end
      %% update C
       C_k  = C + (bk_1 - 1)/bk * (C - C_1);
       Gc_k = C_k - 1/Lip * gradientOfC(YTY,XTY,Z,W,C_k, lambda1,lambda4);
       C_1  = C;
       C    = softthres(Gc_k,lambda3/Lip); 
       C    = max(C,0);
       
      %% update W
       W_k  = W + (bk_1 - 1)/bk * (W - W_1);
       Gw_x_k = W_k - 1/Lip * gradientOfW(XTX,XTY,W_k, C ,Z);
       W_1  = W;
       W    = softthres(Gw_x_k,lambda2/Lip);
       
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
      
       %% update Z
       Z_k  = Z + (bk_1 - 1)/bk * (Z - Z_1);
       Gz_k = Z_k - 1/Lip * gradientOfZ(XTX,XTY,Z,W,C);
       Z_1  = Z;
       Z    = softthres(Gz_k,lambda5/Lip);

      %% Loss
       LS = X*(W+Z) - Y*C;
       DiscriminantLoss = trace(LS'* LS);
       LS = Y*C - Y;
       CorrelationLoss  = trace(LS'*LS);
       CorrelationLoss2 = trace(Y*C*L*(Y*C)');
       sparesW    = sum(sum(W~=0));
       sparesC    = sum(sum(C~=0));
       sparesZ    = sum(sum(Z~=0));
       totalloss = DiscriminantLoss + lambda1*CorrelationLoss + lambda2*sparesW + lambda3*sparesC+lambda4*CorrelationLoss2 + lambda5*sparesZ/2;
       loss(iter,1) = totalloss;
       if abs((oldloss - totalloss)/oldloss) <= miniLossMargin
           break;
       elseif totalloss <=0
           break;
       else
           oldloss = totalloss;
       end
       iter=iter+1;
    end
    model.W = W;
    model.C = C;
    model.Z = Z;
    model.loss = loss;
    model.optmParameter = optmParameter;
end

%% soft thresholding operator
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0);  
end

function gradient = gradientOfW(XTX,XTY,W,C,Z)
    gradient = XTX*(W+Z) - XTY*C;
end

function gradient = gradientOfC(YTY,XTY,Z,W,C, lambda1,lambda4)
    L = diag(sum(C,2)) - C;
    gradient = (lambda1+1)*YTY*C - XTY'*(W+Z) - lambda1*YTY+ lambda4*YTY*C*(L + L');
end

function gradient = gradientOfZ(XTX,XTY,Z,W,C)
    gradient = XTX*(W+Z) - XTY*C;
end

function F_v = calculateF(W, XTX, XTY, YTY, C, lambda1, lambda4,Z)
% calculate the value of function F(\Theta)
    F_v = 0;
    L = diag(sum(C,2)) - C;
    F_v = F_v + 0.5*trace((W+Z)'*XTX*(W+Z)-2*(W+Z)'*XTY*C + C'*YTY*C);
    F_v = F_v + 0.5*lambda1*trace(C'*YTY*C - 2*YTY*C + YTY);
    F_v = F_v + lambda4*trace((YTY*C)'*L*YTY*C);
end

function QL_v = calculateQ(W, XTX, XTY, YTY, C, lambda1, lambda4, Lip,W_t,Z)
% calculate the value of function Q_L(w_v,w_v_t)
    QL_v = 0;
    QL_v = QL_v + calculateF(W_t, XTX, XTY, YTY, C, lambda1, lambda4,Z);
    QL_v = QL_v + 0.5*Lip*norm(W - W_t,'fro')^2;
    QL_v = QL_v + trace((W - W_t)'*gradientOfW(XTX,XTY,W_t,C,Z));
end
