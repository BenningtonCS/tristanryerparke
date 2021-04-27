import torch
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


x = torch.Tensor([[23,6,34],[5,-2,9],[0,1,0]])
kern = torch.Tensor([[0.25,-0.25],[0.5,-0.5]])

print(x,kern)