# Multi-Class Classification with Dropout #

This code is mosty derived from the pytorch quckstart tutorial, but updated to include dropout.

The network consists of these layers:

    * nn.Linear(28*28, 256)c \n
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


