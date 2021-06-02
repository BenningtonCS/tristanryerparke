import numpy as np
from torch import nn
import torch

x = torch.tensor([[3,2,1,2,0,7,12,35],
                  [4,6,9,3,1,0, 0, 0]],
                  dtype=torch.long)




print(x)
print(x.shape)

vocab_size = 1500
hidden_size = 10

emb = nn.Embedding(vocab_size,8,padding_idx=0)
rnn = nn.RNN(8,hidden_size,batch_first=True)
lin =  nn.Linear(hidden_size,4)


mf = emb(x)
mf = mf.to(dtype=torch.long)
print(mf.dtype)
print(mf)

mf2 = rnn(mf)
print(mf2)
