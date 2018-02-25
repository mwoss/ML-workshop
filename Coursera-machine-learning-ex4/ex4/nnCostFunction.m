function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



%Part 1:
X = [ones(m,1) X]; %5000x401, bias unit added
hypot1 = sigmoid(X * Theta1'); %5000x25
hypot1 = [ones(m,1) hypot1]; %5000x26, bias added
hypot_final = sigmoid(hypot1 * Theta2'); %5000x10 matrix

yv = [1:num_labels] == y; % super smart way to create logical matrix described in excercise found in coursera resources
for i=1:num_labels
  J += 1/m * sum(-yv(:,i)' * log(hypot_final(:,i)) - (1 - yv(:,i))' * log(1 - hypot_final(:,i)));
end
%with regularization
J += (lambda/(2*m) * (sum(sum(Theta1(:,2:end) .^2, 1), 2) + sum(sum(Theta2(:,2:end) .^2, 1) , 2)));

%--------------------------------------------------------------------------------------------------


%Part 2:
for i=1:m
  a_1 = X(i,:); % already have bias unit
  
  z_2 = Theta1 *  a_1'; 
  a_2 = sigmoid(z_2);
  a_2 = [1; a_2]; % add bias unit
  
  a_3 = sigmoid(Theta2 * a_2);
  
  delta_3 = a_3 - yv(i, :)'; %setting delta^3_k = a^3_k - y_k
  
  delta_2 = Theta2' * delta_3 .* sigmoidGradient([1; z_2]); % calculate delta_2
  delta_2 = delta_2(2:end); %strip bias
  
  Theta1_grad += delta_2 * a_1; %accumulate gradeints
  Theta2_grad += delta_3 * a_2';
end
Theta1_grad = Theta1_grad ./ m; %obtain unregularized gradient
Theta2_grad = Theta2_grad ./ m;

%--------------------------------------------------------------------------------------------------


%Part 3:
Theta1_grad(:,2:end) += (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:,2:end) += (lambda/m) * Theta2(:, 2:end);
%Regularization, ommiting when j=0 (bias units)



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
