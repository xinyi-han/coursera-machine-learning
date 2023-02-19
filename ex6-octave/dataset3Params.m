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

param = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
num = length(param);
pairs = zeros(num^2, 2); % C, sigma
matrix = param * ones(1, num);
Cs = matrix';
pairs(:, 1) = Cs(:);
pairs(:, 2) = matrix(:);
errors = zeros(num^2, 1);

for i = 1:numel(matrix)
    C = pairs(i, 1);
    sigma = pairs(i, 2);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    errors(i) = mean(double(predictions ~= yval));
end

[~, i] = min(errors);
C = pairs(i, 1);
sigma = pairs(i, 2);

% =========================================================================

end
