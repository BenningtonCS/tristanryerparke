import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
# label data
labels_names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

# load data
training_data = datasets.FashionMNIST(root='data', train=True, download = True, transform=ToTensor())

# Show single image
figure = plt.figure(figsize=(10, 10))
img, label = training_data[1] 
# Change the 1 to a new number for a new image!

# Set up a kernel for detecting vertical edges
vK = torch.tensor([[-1.0, 0, 1.0]])
vertical_edges = cross_correlation(img.squeeze(), vK, padding=(0,1))

# Max/avg pooling with padding and stride
max_pool = pooling(img.squeeze(), (3, 3), 'max', padding = (1, 1))
shrink = pooling(img.squeeze(), (3, 3), 'avg', padding = (1, 1), stride=(2, 2))

figure.add_subplot(2, 2, 1) # Upper left
plt.title(labels_names[label] + " " + str(img.squeeze().shape))
plt.axis("off") #removes pixel labels
plt.imshow(img.squeeze(), cmap="gray")

figure.add_subplot(2, 2, 2) # Upper right
plt.title('vertical' + " " + str(vertical_edges.shape))
plt.axis("off")
plt.imshow(vertical_edges, cmap="gray")

figure.add_subplot(2, 2, 3) # Lower left
plt.title('max pooling' + " " + str(max_pool.shape))
plt.axis("off")
plt.imshow(max_pool, cmap="gray")

figure.add_subplot(2, 2, 4) # Lower right
plt.title('shrunk' + " " + str(shrink.shape))
plt.axis("off")
plt.imshow(shrink, cmap="gray")

plt.show()