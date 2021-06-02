import torch
from torch.utils.data import Dataset, DataLoader
device = 'cpu'

features, labels, vocab_size = torch.load(open("data/trainData.tch",'rb'))

class HeadlinesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

train_data = HeadlinesDataset(features,labels)
train_loader = DataLoader(train_data,shuffle=1,batch_size=64)

class RNN(torch.nn.Module):
    def __init__(self,vocab_size=0,hidden_size=0):
        super(RNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size,8,padding_idx=0)
        self.RNN = torch.nn.RNN(8,hidden_size,batch_first=True)
        self.linear = torch.nn.Linear(hidden_size,4)
        
    def forward(self, features):
        result = self.embedding(features)
        print("after embed:",result.shape)
        result, hidden = self.RNN(result)
        print("hidden:",hidden.shape)
        label = self.linear(hidden[0])
        print("label:",label.shape)
        return label

def train_RNN(model, train_loader, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lossFunc = torch.nn.CrossEntropyLoss()
    for i in range(epochs):
        
        model.train()
        loss = 0

        for X, Y in train_loader:
            optimizer.zero_grad()

            X = X.to(device)
            y_true = Y.to(device)
            y_hat = model(X) 
            #print("hat:",y_hat,"true:",y_true)
            loss = lossFunc(y_hat, y_true) 
            loss += loss.item() * X.size(0)

            loss.backward()
            optimizer.step()

        epoch_loss = float(loss / len(train_loader.dataset))
        print("Epoch Loss:",epoch_loss)

Recurrent = RNN(vocab_size=vocab_size,hidden_size=3)
train_RNN(Recurrent, train_loader, epochs=3)

