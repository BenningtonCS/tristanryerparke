import pandas as pd
import numpy as np
import math

def add_exponents(x,num=1):
    """
    Function to add a column of ones to the beginning of a pandas dataframe,
    and optionally add columns to the end of the original data which are made up of
    the data to the power of a number defined by "num".
    """
    x.insert(0,"ones",np.ones(x.size),True)
    for i in range(2,num+1):
        x.insert(i,"XtoThe{0}".format(i),np.array(x.iloc[:,1] ** i),True)
    return x

def gradient_descent(x, y, step_size, tolerance, max_iterations):
    """Function to perform gradient descent on a dataset of x trying to predict y.
    x must be at minimum a two-dimensional numpy array with the first column made up of ones.
    y must be at minimum a one-dimensional numpy array with the same number of rows as x.
    """
    iterations = 0 
    magnitude = float('inf')
    w = np.zeros(x.shape[1])
    derivatives = np.zeros(x.shape[1])
    
    while True:
        y_hat = np.matmul(x,w)
        errors = np.subtract(y_hat,y)

        for i in range(0,w.shape[0]):
            derivatives[i] = sum(np.multiply(errors,x[:,i]))
            w[i] = w[i] - derivatives[i] * step_size

        magnitude = math.sqrt(np.dot(derivatives,derivatives))

        rss = sum(errors ** 2)

        iterations = iterations + 1
        
        #exit loop?
        if magnitude <= tolerance:
            break
        if iterations >= max_iterations:
            break
    
    print("Summary:","\n")
    print("# of iterations: ",iterations)
    print("Final Weights: ",w)
    print("Magnitude of Gradient: ",magnitude)
    #print("RSS: ",rss)
    print("MSE: ",rss/x.shape[0])

def divide_data(data, training_percent, validation_percent, shuffle):
    #print(data.shape[0]*training_percent)
    #print(data.shape[0]*validation_percent)

    training_size = math.floor(data.shape[0]*training_percent)
    validation_size = math.floor(data.shape[0]*validation_percent)
    test_size = data.shape[0] - training_size - validation_size
    
    training_data = data.iloc[:training_size,:]
    validation_data = data.iloc[training_size : training_size + validation_size, : ]
    test_data = data.iloc[training_size + validation_size + 1 :,:]

    print(training_data.shape)
    print(validation_data.shape)
    print(test_data.shape)
    
    training_data.to_csv(r'Machine Learning S21/training_data.csv')
    validation_data.to_csv(r'Machine Learning S21/validation_data.csv')
    test_data.to_csv(r'Machine Learning S21/test_data.csv')