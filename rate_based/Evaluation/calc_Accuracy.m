%%%
%Path to the library of the classifier must be adjusted. #Line 44/52
%Path to the labels must be adjusted, if not already loaded. #Line 62/66
%%%
%% Init
layer_list={'LGN','V1_L4_Exc','V1_L4_Inh','V1_L23_Exc','V1_L23_Inh','V2_L4_Exc','V2_L4_Inh','V2_L23_Exc','V2_L23_Inh'};

suffix='t42d_';

num_threads=4;
search_param=0;

%% Save results as text
% Open file for saving results
fid = fopen(['Accuracies',suffix,replace((datestr(datetime)),' ','_'),'.txt'],'wt');

if exist('lib','var')~=1
    %Select libSVM 0 or liblinear 1
    %Lib linear as standard for speed reasons
    lib=1;
end

% Try to set OMP environment variables
system('export OMP_DYNAMIC=true');
system(['export OMP_NUM_THREADS=',num2str(num_threads)]);

if lib==1
    % Linear SVM with liblinear | Sometimes suboptimal results, but fast and parallel
    % -s
%    0 -- L2-regularized logistic regression (primal)
% 	 1 -- L2-regularized L2-loss support vector classification (dual)
% 	 2 -- L2-regularized L2-loss support vector classification (primal)
% 	 3 -- L2-regularized L1-loss support vector classification (dual)
% 	 4 -- support vector classification by Crammer and Singer
% 	 5 -- L1-regularized L2-loss support vector classification
% 	 6 -- L1-regularized logistic regression
% 	 7 -- L2-regularized logistic regression (dual)
    
%     SVM_param=['-s 1 -B 1 -q -n ',num2str(num_threads)];%L2-regularized L2-loss support vector classification (dual), Bias enabled, quiet, # threads
    SVM_param=['-s 1 -B 1 -q -m ',num2str(num_threads)];
    fprintf('Start SVM classification with liblinear\n')
    
    %Add path to liblinear for Matlab
    addpath /PATH_TO_LIBLINEAR/MATLAB/liblinear-multicore-2.42/matlab
else
    % Linear SVM with libSVM | Slow but gives best results
    % Linear SVM
    SVM_param='-t 0 -q -m 1000';%linear kernel, quiet, 1000mb cache
    fprintf('Start SVM classification with linear SVM (libSVM)\n')
    
    %Add path to libsvm for Matlab
    addpath /PATH_TO_LIBSVM/MATLAB/libsvm-3.21/matlab
end

%% MNIST
fprintf(fid,'-----MNIST-----\n');
fprintf(fid,[SVM_param,'\n']);

%% Load labels
if exist('test_labels','var')==0
    # Path to the lables
    load('~/DATA/MNIST/MNIST_test_labels.mat')
end
if exist('train_labels','var')==0
    # Path to the lables
    load('~/DATA/MNIST/MNIST_train_labels.mat')
end

acc=zeros(numel(layer_list),1);

%% Calculate accuracies
for i=1:numel(layer_list)
    fprintf([layer_list{i},'\n'])
    % Load responses
    r_test = permute(hdf5read(['./test_responses_',suffix,layer_list{i},'.h5'],['r_',layer_list{i}]),[4,3,2,1]);
    r_train = permute(hdf5read(['./train_responses_',suffix,layer_list{i},'.h5'],['r_',layer_list{i}]),[4,3,2,1]);
    
    % Train SVM
    if search_param==1
        c_value=train(train_labels,sparse(double(r_train(:,:))),[SVM_param,' -C']);
    else
        c_value=1;
    end
    svmstruct=train(train_labels,sparse(double(r_train(:,:))),[SVM_param,' -c ', num2str(c_value(1))]);
    [predicted_label, accuracy, d_p] = predict(test_labels, sparse(double(r_test(:,:))), svmstruct);
    fprintf(fid,'%s %f c=%i\n',layer_list{i},accuracy(1),c_value(1));
    acc(i)=accuracy(1);
end

save(['accuracies',suffix,'.mat'],'acc')

%Close text file
fclose(fid);
