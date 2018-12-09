function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
params = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
p_size = size(params,1);

predicted_err = zeros(p_size, p_size);

for i=1:p_size
  for j=1:p_size
    model = svmTrain(X, y, params(i), @(x1, x2)gaussianKernel(x1,x2,params(j)), 1e-3, 5);
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    predicted_err(i,j) = error;
  end
end

[values_1, row_indecies] = min(predicted_err);
[values_2, j_index] = min(values_1);

C = params(row_indecies(j_index));
sigma = params(j_index);





% =========================================================================

end
