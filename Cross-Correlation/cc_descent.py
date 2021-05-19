import torch
from torchvision import datasets
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from manual_functions import cross_correlation
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

def gradientDescent(parameters, step_size):
    with torch.no_grad():
        for parameter in parameters:
            #print(parameters,"\n")
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
    return kernel, bias

loss_func = torch.nn.MSELoss() 
kernel, bias = train(x,y,1e-2,10)
print("Kernel: {0}, Bias: {1}".format(kernel, bias))





