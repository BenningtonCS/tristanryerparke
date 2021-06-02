# Lenet 5 Pytorch experimentation #

Five files, each based off of this tutorial:
https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320

* lenet-ish_1.py uses the ReLu activation function.
* lenet-ish_2.py uses the TanH activation function.
* lenet-ish_3.py uses the LeakyReLu activation function.
* lenet-ish_4.py has two dropout layers after the first two convolutions, with p values 0.5 and 0.2
* lenet-ish_5.py uses the sigmoid activation function.

With the same hyperparameters, similar performance was achived at 15 epochs.
Training and Validation loss barely changed for each model.
Relu activation saeemed to perform the best, with leaky relu giving similar results.
The dropout layers in #4had a slight negative effect on the accuracy of the model.
Sigmoid activation function really seemed to make the results worse.

1: Train accuracy: 92.06   Valid accuracy: 89.70

2: Train accuracy: 92.27   Valid accuracy: 88.48

3: Train accuracy: 92.00   Valid accuracy: 89.54

4: Train accuracy: 90.52   Valid accuracy: 88.56

5: Train accuracy: 87.83   Valid accuracy: 86.35

Training #1 for 50 epochs produced a much better training accuracy, but was less accuate on the valudation than 15 epochs.
Train accuracy: 95.27   Valid accuracy: 89.00

It seems as if #1 trained at 15 epochs performs the best amongst all the variations included.

