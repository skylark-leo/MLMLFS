
addpath(genpath('.'));
clc
starttime = datestr(now,0);                           
deleteData  = 0;
model_LSML.misRate = 0.8; % missing rate of positive  class labels
[optmParameter, modelparameter] =  initialization;% parameter settings for LSML
model_LSML.optmParameter = optmParameter;
model_LSML.modelparameter = modelparameter;
model_LSML.tuneThreshold = 1;% tune the threshold for mlc
fprintf('*** run myMLAL for multi-label learning with missing labels ***\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load the dataset, you can download the other datasets from our website
load('recreation.mat');
if exist('train_data','var')==1
    data    = [train_data;test_data];
    target  = [train_target,test_target];
    if deleteData == 1
        clear train_data test_data train_target test_target
    end
end
data      = double (data);
num_data  = size(data,1);
temp_data = data + eps;
temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
if sum(sum(isnan(temp_data)))>0
    temp_data = data+eps;
    temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
end
temp_data = [temp_data,ones(num_data,1)];
randorder = randperm(num_data);
cvResult  = zeros(16,modelparameter.cv_num);
%%
for i = 1:modelparameter.repetitions       
    for j = 1:modelparameter.cv_num
        fprintf('- Repetition - %d/%d,  Cross Validation - %d/%d', i, modelparameter.repetitions, j, modelparameter.cv_num);
        [cv_train_data,cv_train_target,cv_test_data,cv_test_target ] = generateCVSet( temp_data,target',randorder,j,modelparameter.cv_num );

        if model_LSML.misRate > 0
             temptarget = cv_train_target;
             [ImcompleteTarget, ~, ~, realpercent]= getIncompleteTarget(cv_train_target, model_LSML.misRate,1); 
             fprintf('\n-- Missing rate:%.1f, Real Missing rate %.3f\n',model_LSML.misRate, realpercent); 
        end
       %% Training
        modelLSML  = MyMLAL( cv_train_data, ImcompleteTarget,optmParameter); 
       %% Prediction and evaluation
        Outputs = (cv_test_data*modelLSML.W)';
        if model_LSML.tuneThreshold == 1
            fscore                 = (cv_train_data*modelLSML.W)';
            [ tau,  currentResult] = TuneThreshold( fscore, cv_train_target', 1, 2);
            Pre_Labels             = Predict(Outputs,tau);
        else
            Pre_Labels = double(Outputs>=0.5);
        end
        
        %%fprintf('-- Evaluation\n');
        tmpResult = EvaluationAll(Pre_Labels,Outputs,cv_test_target');
        cvResult(:,j) = cvResult(:,j) + tmpResult;
    end
end
endtime = datestr(now,0);
cvResult = cvResult./modelparameter.repetitions;
Avg_Result      = zeros(16,2);
Avg_Result(:,1) = mean(cvResult,2);
Avg_Result(:,2) = std(cvResult,1,2);
model_LSML.avgResult = Avg_Result;
model_LSML.cvResult  = cvResult;
PrintResults(Avg_Result);