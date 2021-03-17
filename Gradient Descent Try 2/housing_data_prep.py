import GD_Utils as tml
import numpy as np
import pandas as pd
import math

def divide_data(data, training_percent, validation_percent, shuffle):
    """
    Function to devide a pandas dataFrame into three datasets with the option to shuffle.
    Also writes the datasets to csv
    """
    #Shuffle
    if shuffle:
        print("Shuffling data...")
        np.random.seed(42)
        data.reindex(np.random.permutation(data.index)) 

    #Calculate integer size of each dataset
    training_size = math.floor(data.shape[0]*training_percent) 
    validation_size = math.floor(data.shape[0]*validation_percent) 
    test_size = data.shape[0] - training_size - validation_size 
    
    #Split up the datasets
    print("Spliting the data...")
    training_data = data.iloc[:training_size,:] 
    validation_data = data.iloc[training_size : training_size + validation_size, : ]
    test_data = data.iloc[training_size + validation_size + 1 :,:]

    #Check sizes
    print("The split datasets have these sizes:")
    print(training_data.shape)
    print(validation_data.shape)
    print(test_data.shape)

    #Write the files
    training_data.to_csv(r'Datasets/training_data.csv')
    validation_data.to_csv(r'datasets/validation_data.csv')
    test_data.to_csv(r'datasets/test_data.csv')
    print("Wrote datasets to csv.")

#Read the dataset
housing_raw_data = pd.read_csv("Real_Estate_Sales_730_Days.csv")

#Clean it up a little
housing_data = housing_raw_data[(housing_raw_data['TotalAppraisedValue'] < 2e6 ) & (housing_raw_data['TotalFinishedArea'].notna())]

#Grab the columns we care about
housing_data = housing_data[['TotalFinishedArea','TotalAppraisedValue']]

#Devide and write the data
divide_data(housing_data,0.8,0.1,shuffle=True)

