# Linear Regression manual version
import numpy as np
import torch
from torch.utils import data

def synthetic_data(w : torch.tensor, b : float, num_examples : int):
    #Create data from normal dist based on given weights
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
    
def minibatch_iterator(inputs, output, batch_size, shuffle=True):
    dataset = data.TensorDataset(inputs, output)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle)
    
def linear_regression_model(X, weights, bias):
    return X @ weights + bias
    
def squared_loss(y_hat, y):
    return (y_hat - y)**2 / 2

def l2_loss(w,l2_lambda):
    return l2_lambda * sum(w ** 2)

#add additional loss function that sums squares of w and add on line 39
    
def gradient_descent(parameters, step_size, batch_size):
    with torch.no_grad():
        for param in parameters:
            param -= (param.grad * step_size/batch_size)
            param.grad.zero_()

def train(X, Y, step_size=0.05, batch_size=1, num_epochs=0, l2_lambda=1e-5):
    m = X.shape[1]
    weights = torch.normal(0, 0.01, size=(m, 1), requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    
    for epoch in range(num_epochs):
        for mini_X, mini_Y in minibatch_iterator(X, Y, batch_size, True):
            Y_hat = linear_regression_model(mini_X, weights, bias)
            loss = squared_loss(Y_hat, mini_Y) + l2_loss(weights, l2_lambda)
            loss.sum().backward()
            gradient_descent([weights, bias], step_size, batch_size)
        with torch.no_grad():
            y_hat = linear_regression_model(X, weights, bias)
            training_loss = squared_loss(y_hat, Y)
            print('epoch', epoch, 'loss', training_loss.mean())
    return weights, bias

#Initailize the data with weights= [-2.3, 1.8, 3.6] and bias= 5.2
true_w = torch.tensor([-2.3, 1.8, 3.6])
features, labels = synthetic_data(true_w, 5.2, 1000)

##weights= [-2.3, 1.8, 3.6] and bias= 5.2
#If the trained model weights match those given above, the model is converging correctly.
weights, bias = train(features, labels, step_size=0.03, batch_size = 10, num_epochs=10, l2_lambda=1e-5)

print('bias', bias)
print('weights', weights)