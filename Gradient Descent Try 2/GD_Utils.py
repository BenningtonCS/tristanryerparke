import pandas as pd
import numpy as np
import math



def gradient_descent(x_dataFrame, y_dataFrame, tolerance, max_iterations,step_size=None, debug=False):
    """Function to perform gradient descent on a dataset of x trying to predict y.
    x must be at minimum a two-dimensional numpy array with the first column made up of ones.
    y must be a one-dimensional numpy array with the same number of rows as x.
    """
    x_dataFrame.insert(0,"ones",np.ones(x_dataFrame.shape[0]),True)

    x = np.array(x_dataFrame)
    y = np.array(y_dataFrame.iloc[:, 0])
    
    iterations = 0 
    magnitude_sq = float('inf')
    w = np.zeros(x.shape[1])
    derivatives = np.zeros(x.shape[1])

    print("Training...")
    
    while True:
        y_hat = np.matmul(x,w)
        errors = np.subtract(y_hat,y)

        for i in range(0,w.shape[0]):
            derivatives[i] = sum(np.multiply(errors,x[:,i]))

        magnitude_sq = np.dot(derivatives,derivatives)
        alpha = step_size
        if step_size == None:
            alpha = 1
            beta = 0.8

            def J(x, y, w):
                errors_J = x @ w - y
                return 0.5 * sum(errors_J ** 2) 
            
            new_w = w - alpha * derivatives
            
            left = J(x, y, new_w)
            right = J(x, y, w) - alpha * magnitude_sq * 0.5

            while left - right >= 1e-20:
                #print("difference = ",left - right)
                alpha *= beta
                new_w = w - alpha * derivatives
                left = J(x, y, new_w)
                right = J(x, y, w) - alpha * magnitude_sq * 0.5

        rss = sum(errors ** 2)

        iterations += 1
        
        for i in range(0,w.shape[0]):
            w[i] = w[i] - derivatives[i] * alpha
        if debug == True:
            print("step_size = ",alpha)
            print("Iteration ", iterations, ", Weights =",w)
        
        if magnitude_sq <= tolerance ** 2:
            break
        if iterations >= max_iterations:
            break
    
    
    output = {"iterations":iterations, "weights":w, "MSE":rss/x.shape[0], "y_hat":y_hat}
    print(output)
    return output

def add_polynomials(x,num=1):
    """
    Function to optionally add columns to the end of the original data which are made up of
    the data to the power of a number defined by "num".
    """
    #for each exponent specified by num, add a column with the data 
    for i in range(2,num+1):
        column_name = "XtoThe{0}".format(i)
        column_contents = x.iloc[:,0] ** i
        x.insert(i-1,column_name,column_contents,True)
    return x

def normalize(input_data,input_bounds=None):
    
    normalized_data = input_data.copy()
    print("Data has these columns: ",normalized_data.columns)
    column_bounds = []

    for i, column in enumerate(input_data.columns):
        #see if there are external min/max values
        if input_bounds != None:
            data_min = input_bounds[i][0]
            data_max = input_bounds[i][1]
        else:
            data_min = normalized_data[column].min()
            data_max = normalized_data[column].max()
        normalized_data[column] = (input_data[column] - data_min) / (data_max - data_min)
        column_bounds.append((data_min,data_max))
    
    return normalized_data, column_bounds

def un_normalize(input_data,orig_bounds):
    print("Un-normalizing")

    data_min = orig_bounds[0][0]
    data_max = orig_bounds[0][1]

    un_data = (input_data * (data_max - data_min)) + data_min
    column_bounds = (data_min,data_max)
    print("new bounds = ",column_bounds)
    
    return un_data
