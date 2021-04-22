# Linear Regression concise version
import numpy as np
import torch
from torch.utils import data

def synthetic_data(w, b, num_examples):
    #Create data from normal dist based on given weights
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
    
def minibatch_iterator(inputs, output, batch_size, shuffle=True):
    dataset = data.TensorDataset(inputs, output)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle)

def train(net, optimizer, loss, data_iterator, num_epochs = 1):
    loss_sum = 0
    total_samples = 0
    
    for epoch in range(num_epochs):
        for mini_X, mini_Y in data_iterator:
            Y_hat = net(mini_X)
            l = loss(Y_hat, mini_Y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_sum += l
            total_samples += mini_X.shape[0]

#Initailize the data with weights= [-2.3, 1.8, 3.6] and bias= 5.2
true_w = torch.tensor([-2.3, 1.8, 3.6])
features, labels = synthetic_data(true_w, 5.2, 1000)

#Define the neural network
linear = torch.nn.Linear(features.shape[1], 1)
linear.weight.data.normal_(0, 0.01) #weights = torch.normal(0, 0.01, size=(m, 1), requires_grad=True)
linear.bias.data.fill_(0) #bias = torch.zeros(1, requires_grad=True)

#Declare L2 Lambda value
l2_lambda = 1-0.9

#Declare optimizer, loss and data iterator
optimizer = torch.optim.SGD(linear.parameters(), lr=0.03, weight_decay=l2_lambda)
squared_loss = torch.nn.MSELoss() 
data_iterator = minibatch_iterator(features, labels, 10, True)

#weights= [-2.3, 1.8, 3.6] and bias= 5.2
#If the trained model weights match those given above, the model is converging correctly.
train(linear, optimizer, squared_loss, data_iterator, num_epochs=3)

print('bias', linear.bias)
print('weights', linear.weight)