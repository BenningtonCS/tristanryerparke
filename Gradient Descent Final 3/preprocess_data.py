import GD_Utils as gdu
import numpy as np
import pandas as pd
import math
import pickle

raw_data = pd.read_csv('datasets/Real_Estate_Sales_730_Days.csv')
raw_data = raw_data[(raw_data['TotalAppraisedValue'] < 2e6 ) & (raw_data['TotalFinishedArea'].notna())]
data = raw_data[['TotalFinishedArea','TotalAppraisedValue']]
data, bounds = gdu.normalize(data,maxMin=True)

data = data.sample(frac=1,random_state=3)

training_size = int(data.shape[0] * 0.8)
validation_size = int(data.shape[0] * 0.1)

training_data = data.iloc[:training_size,:] 
validation_data = data.iloc[training_size :training_size + validation_size,:] 
test_data = data.iloc[training_size + validation_size :,:] 

print(training_data)
print(test_data)

training_data.to_csv(r'Datasets/training_data.csv')
validation_data.to_csv(r'datasets/validation_data.csv')
test_data.to_csv(r'datasets/test_data.csv')
pickle.dump(bounds,open( "datasets/bounds.p", "wb" ))




