import torch


f = torch.torch.tensor([[0.0, 1.0, 2.0], 
                        [3.0, 4.0, 5.0], 
                        [6.0, 7.0, 8.0]])

g = torch.tensor([[-1.0, 1.0], 
                  [2.0, 0.5]])

x, y  = f.shape

u, v = g.shape 

out_x, out_y = x - u + 1, y - v + 1
print("output size: ",out_x,":",out_y)

output = torch.zeros((out_x,out_y))

for i in range(0,out_x):
    for j in range(0,out_y):
        print(i,":",j)
        sub_mat = f[i : u+i, j : v+j]
        print(sub_mat)
        output[i,j] = torch.sum(sub_mat * g)

print(output)