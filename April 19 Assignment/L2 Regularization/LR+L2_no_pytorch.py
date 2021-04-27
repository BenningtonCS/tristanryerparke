import numpy as np
np.set_printoptions(suppress=True)


def synthetic_data(w, b, num_examples):
    #Create data from normal dist based on given weights
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y +=  np.random.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def linear_regression_step(x, y, weights, bias, l2_lambda, step_size):
    #Executes one step of linear regression
    y_hat = x @ weights + bias
    errors = y_hat - y
    derivatives = np.zeros(shape=(weights.shape[0],1))
    for i in range(0, len(weights)):
        derivatives[i] = l2_lambda * sum(errors * np.expand_dims(x[:,i],1))
        weights[i] -= derivatives[i] * step_size
    bias_deriv = l2_lambda * sum(errors * np.expand_dims(bias,1))
    bias = bias - bias_deriv * step_size
    return weights, bias


def train(x, y, step_size=0.03, batch_size=10, l2_lambda=0.8, num_epochs=1):
    #Trains an LR model with L2 Regularization
    phi = x.shape[1]
    weights = np.random.normal(0, 0.01, size=(phi,1))
    bias = np.ones(1)
    
    batches_x = np.vsplit(x,len(features)/batch_size)
    batches_y = np.vsplit(y,len(features)/batch_size)

    for epoch in range(num_epochs):
        print("epoch ",epoch," weights = ",weights," bias = ",bias)
        for section_x, section_y in zip(batches_x, batches_y):
            weights, bias = linear_regression_step(section_x, section_y, weights, bias, l2_lambda, step_size)

#Initailize the data with weights= [-2.3, 1.8, 3.6] and bias= 5.2
true_w = np.array([-2.3, 1.8, 3.6])
features, labels = synthetic_data(true_w, 5.2, 1000)

#weights= [-2.3, 1.8, 3.6] and bias= 5.2
#If the trained model weights match those given above, the model is converging correctly.
train(features, labels, step_size=0.03, batch_size=10, l2_lambda=0.9, num_epochs=3)


