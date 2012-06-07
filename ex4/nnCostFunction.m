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

% Part 1: Cost function

A1= [ones(size(X, 1), 1) X];
Z2= A1 * Theta1' ;
A2= sigmoid(Z2);
A2= [ones(size(A2, 1), 1) A2];
Z3= A2 * Theta2' ;
A3= sigmoid(Z3);

ymat = zeros(m, num_labels);

for i= 1:m
	ymat(i, y(i)) = 1;
end

cost= (-ymat .* log(A3)) - ((1-ymat) .* (log(1-A3))) ;
cost= sum(sum(cost, 2))/m;


% Part 2: Backpropagation
#{
d3 = A3 - ymat;

%d2= (d3 * Theta2) .* A2 .* (1-A2);
d2= (d3 * Theta2) .* sigmoidGradient(A2);
d2= d2(:,2:size(d2,2));

accum2= A2' * d3;
accum2_n = size(accum2, 2);

reg2 = Theta2';
reg2(1,:) = zeros(1, accum2_n);
reg2 = (lambda/m) * reg2;

accum2= accum2/m + reg2;

% accum2= accum2(2:size(accum2, 1));
% accum2 = sum(accum2);

d1 = (d2 * Theta1) .* sigmoidGradient(A1);

accum1= A1' * d2;
accum1_n = size(accum1, 2);

reg1 = Theta1';
reg1(1,:) = zeros(1, accum1_n);
reg1 = (lambda/m) * reg1;
accum1= accum1/m + reg1;
% accum1= accum1(2:size(accum1, 1));
% accum1 = sum(accum1);

Theta1_grad = accum1;
Theta2_grad = accum2;
% Part 3: Calculate regularization
reg= 0;
for i= 1:size(Theta1, 1)
	for j= 2:size(Theta1,2)
		reg= reg + (Theta1(i,j) ^ 2);
	end
end
	
for i= 1:size(Theta2, 1)
	for j= 2:size(Theta2,2)
		reg= reg + (Theta2(i,j) ^ 2);
	end
end
reg
reg= reg * lambda * (1/(2* m));
#}
J = cost + reg;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
