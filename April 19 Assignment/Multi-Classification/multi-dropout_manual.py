import torch
from torch._C import is_autocast_enabled
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
#torch.set_printoptions(sci_mode=False)

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
#Loss function
def CrossEntropyLoss(yHat, y):
    one_hot = torch.zeros(yHat.shape)
    y = y.unsqueeze(1)
    one_hot.scatter_(1, y, 1) #what is this doing
    neg_log = -torch.log(yHat)
    return (one_hot * neg_log).sum(axis=1) #what is axis=1

def gradientDescent(parameters, step_size, batch_size):
    with torch.no_grad():
        for parameter in parameters:
            parameter -= parameter.grad * step_size / batch_size
            parameter.grad.zero_()

def softmax(score):
    e_x = torch.exp(score)
    e_sum = e_x.sum(axis=1, keepdim=True)
    return e_x / e_sum

def relu(x):
    all_zeros = torch.zeros_like(x)
    return torch.max(all_zeros, x)

def dropout(X, prob):
    rand = torch.Tensor(X.shape).uniform_(0,1)
    mask = rand > prob
    return mask.float() * X / (1-prob)

def softmaxRegression(X , w_hid1, b_hid1, w_hid2, b_hid2, w_out, b_out, drop_hid1, drop_hid2, is_training=True):
    #Most of confusion is about the order of things here and why
    flattened = X.reshape((-1, w_hid1.shape[0]))
    score = flattened @ w_hid1 + b_hid1
    h1 = relu(score)
    if is_training:
        h1 = dropout(h1, drop_hid1)
    score = h1 @ w_hid2 + b_hid2 
    #these matrix multiplication, why are they still possible when one can change the size of w_h1 and w_h2.
    #what information is being condensed here and how?
    h2 = relu(score)
    if is_training:
        h2 = dropout(h2, drop_hid2)
    score = h2 @ w_out + b_out
    return(softmax(score))
    
def train(num_inputs, num_outputs, data_iterator, num_epochs, num_hid1, num_hid2, step_size):
    
    #Initialize weights
    w_hid1 = torch.normal(0, 0.1, size=(num_inputs, num_hid1), requires_grad=True)
    w_hid2 = torch.normal(0, 0.1, size=(num_hid1, num_hid2), requires_grad=True)
    w_out = torch.normal(0, 0.1, size=(num_hid2, num_outputs), requires_grad=True)
    #Initialize bias
    b_hid1 = torch.zeros(num_hid1, requires_grad=True)
    b_hid2 = torch.zeros(num_hid2, requires_grad=True)
    b_out = torch.zeros(num_outputs, requires_grad=True)

    for epoch in range(0, num_epochs):
        accumulated_loss = 0
        total_samples = 0
        for x, y in data_iterator:
            y_hat = softmaxRegression(x, w_hid1, b_hid1, w_hid2, b_hid2, w_out, b_out, 0.2, 0.5)
            loss = CrossEntropyLoss(y_hat, y)
            loss.sum().backward()
            gradientDescent([w_hid1, b_hid1, w_hid2, b_hid2, w_out, b_out],step_size,x.shape[0])
            accumulated_loss += loss.sum()
            total_samples += x.shape[0]

        print("Epoch ",epoch + 1, " Loss:", accumulated_loss/total_samples)
    return [w_hid1, b_hid1, w_hid2, b_hid2, w_out, b_out]

if __name__ == '__main__':
    
    data_iterator = DataLoader(training_data, 256, shuffle=True)
    weights = train(784, 10, data_iterator, 2, 256, 256 ,0.1)
    #how can we discern the number of inputs from the training data instead of just knowing 784?
    for i, weight in enumerate(weights):
        print("W",i,": ",weight)