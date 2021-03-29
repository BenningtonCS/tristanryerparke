import numpy as np
import pandas as pd
import math

def normalize(in_data,maxMin=True,zScore=False):
    if maxMin == True:
        normalized_data = in_data.copy()
        column_bounds = []

        for i, column in enumerate(in_data.columns):
            data_min = normalized_data[column].min()
            data_max = normalized_data[column].max()
            normalized_data[column] = ( in_data[column] - data_min ) / ( data_max - data_min )
            column_bounds.append((data_min,data_max))
        print("Bounds: ",column_bounds)
        return normalized_data, column_bounds
    else:
        normalized_data = in_data.copy()
        column_mean_std = []

        for i, column in enumerate(in_data.columns):
            data_mean = normalized_data[column].mean()
            data_std = normalized_data[column].std()
            normalized_data[column] = (in_data[column] - data_mean ) / data_std
            column_mean_std.append((data_mean,data_std))
        print("Mean/Std: ",column_mean_std)
        return normalized_data, column_mean_std


def add_polynomial(x,num=1):
    
    for i in range(2,num+1):
        column_name = "XtoThe{0}".format(i)
        column_contents = x.iloc[:,0] ** i
        x.insert(i-1,column_name,column_contents,True)
    return x


def gradient_descent(x_dataFrame, y_dataFrame, tolerance, max_iterations,step_size=None):
   
    x_dataFrame.insert(0,"ones",np.ones(x_dataFrame.shape[0]),True)
    x = np.array(x_dataFrame)
    y = np.array(y_dataFrame)
    
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
        
        #Backtracking Line Search
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

            while left >= right:
                alpha *= beta
                new_w = w - alpha * derivatives
                left = J(x, y, new_w)
                right = J(x, y, w) - alpha * magnitude_sq * 0.5

        rss = sum(errors ** 2)
        iterations += 1
        for i in range(0,w.shape[0]):
            w[i] = w[i] - derivatives[i] * alpha

        if iterations % 10 == 0:
            print(iterations," Iterations")
        if magnitude_sq <= tolerance ** 2:
            break
        if iterations >= max_iterations:
            break
    
    
    output = {"iterations":iterations, "weights":w, "MSE":rss/x.shape[0], "y_hat":y_hat}
    print("weights: ",w)
    print("MSE: ",rss/x.shape[0])
    return output