# Multi-Class Classification with Dropout #

Using multi-droput_concise.py

This code is mosty derived from the pytorch quckstart tutorial, but updated to include dropout.

The network consists of these layers:

    * nn.Linear(28*28, 256)
    Hidden Layer which defines the input image tensor size and a tensor of size 256 as output.
    * nn.ReLU()
    Rectified Linear Unit layer which takes the max of 0 and input, setting all negative values to 0.
    * nn.Dropout(0.2)
    Dropout layer which randomly disables 20% of the nodes in the network, allowing the model to perform well with naturally noisey data.
    * nn.Linear(256, 256)
    Hidden Layer which defines the input tensor size as 256 and output tensor size as 256.
    * nn.ReLU()
    * nn.Dropout(0.5)
    Dropout layer which randomly disables 50% of the nodes in the network.
    * nn.Linear(256, 10)
    Hidden Layer which defines the input tensor size as 256 and output tensor size as 10.

The hyperparameters for this model are the two dropout probabilities, the optimizer learning rate, number of epochs and the batch size.
The model uses a cross entropy loss function and backpropogates the error using stochastic gradient descent.

By changing these hyperparameters and training different models, we can see how the accuracy changes:

Justin's/pytorch tutorial hyperparamters:
Droput 1 = 0.2 Dropout 2 = 0.5 batch_size = 64 epochs = 5 learning rate = 1e-3
Results:
Accuracy: 64.1%, Avg loss: 0.018137 

Same paramters but 15 epochs:
Results:
Accuracy: 74.1%, Avg loss: 0.011130 

Dropout layers commented out and 15 epochs:
Results:
Accuracy: 74.9%, Avg loss: 0.010950 



m

