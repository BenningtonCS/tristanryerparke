import torch
from torchvision import datasets
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
torch.set_printoptions(sci_mode=False)

x = torch.tensor(
       [[1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 0., 0., 1., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 1., 0., 0., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1.]])

y = torch.tensor(
       [[ 0.,  0.,  1.,  1.,  0.,  0.,  0.],
        [ 0.,  1.,  1.,  0.,  0.,  0.,  0.],
        [ 1.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0.,  0., -1., -1.,  0.],
        [ 0.,  0.,  0., -1., -1.,  0.,  0.]])

def cross_correlation(f,g):
    x, y  = f.shape
    u, v = g.shape 

    out_x, out_y = x - u + 1, y - v + 1
    output = torch.zeros((out_x,out_y))
    for i in range(0,out_x):
        for j in range(0,out_y):
            sub_mat = f[i : u+i, j : v+j]
            output[i,j] = torch.sum(sub_mat * g)
    return output

def gradientDescent(parameters, step_size):
    with torch.no_grad():
        for parameter in parameters:
            print(parameters,"\n")
            parameter -= parameter.grad * step_size
            parameter.grad.zero_()

def train(x,y,step_size,num_epochs):
    kernel = torch.normal(0, 0.1 ,size=(2,2),requires_grad=True)
    bias = torch.zeros((1,1),requires_grad=True)
    for epoch in range(0,num_epochs):
        y_hat = cross_correlation(x, kernel) + bias
        loss = loss_func(y_hat,y)
        loss.sum().backward()
        gradientDescent([kernel, bias],step_size)

loss_func = torch.nn.MSELoss() 
train(x,y,1e-2,10)




