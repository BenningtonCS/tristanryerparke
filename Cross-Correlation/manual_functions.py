import torch

f = torch.torch.tensor([[0.0, 1.0, 2.0], 
                        [3.0, 4.0, 5.0], 
                        [6.0, 7.0, 8.0]])

g = torch.tensor([[-1.0, 1.0], 
                  [2.0, 0.5]])


def cross_correlation(f,g,padding=0,stride=1):
    #Pad
    pad_col = torch.zeros((f.shape[0],padding))
    f = torch.hstack((pad_col,f,pad_col))
    pad_row = torch.zeros((padding,f.shape[1]))
    f = torch.vstack((pad_row,f,pad_row))
    #Perform cc
    x, y = f.shape
    u, v = g.shape
    out_x, out_y = x - u + 1, y - v + 1
    output = torch.zeros((out_x,out_y))
    for i in range(0,out_x,stride):
        for j in range(0,out_y,stride):
            #print(i,":",j)
            sub_mat = f[i : u+i, j : v+j]
            #print(sub_mat)
            output[i,j] = torch.sum(sub_mat * g)
    return output

def pooling(f,pool_type,padding=0,stride=1):
    #Pad
    pad_col = torch.zeros((f.shape[0],1))
    f = torch.hstack((pad_col,f,pad_col))
    pad_row = torch.zeros((1,f.shape[1]))
    f = torch.vstack((pad_row,f,pad_row))
    #Perform pooling
    x, y = f.shape
    u, v = g.shape
    out_x, out_y = x - u + 1, y - v + 1
    output = torch.zeros((out_x,out_y))
    for i in range(0,out_x,stride):
        for j in range(0,out_y,stride):
            #print(i,":",j)
            sub_mat = f[i : u+i, j : v+j]
            #print(sub_mat)
            if pool_type == "avg":
                output[i,j] = torch.mean(sub_mat)
            elif pool_type == "max":
                output[i,j] = torch.max(sub_mat)     
    return output

print(cross_correlation(f, g, padding=2,stride=1))
print(pooling(f, pool_type="avg", padding=2,stride=1))

