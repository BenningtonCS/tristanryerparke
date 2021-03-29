import GD_Utils as gdu
import numpy as np
import pandas as pd
import math
import pickle
import matplotlib.pyplot as plt

weights = pickle.load(open( "models/model1.p", "rb" ))

test_data = pd.read_csv("datasets/test_data.csv")

inputs = test_data['TotalFinishedArea'].to_frame()
outputs = test_data['TotalAppraisedValue']

inputs = gdu.add_polynomial(inputs,num=3)

inputs.insert(0,"ones",np.ones(inputs.shape[0]),True)
x = np.array(inputs)
y = np.array(outputs)

test_y_hat = x @ weights

MSE = (sum((test_y_hat - y) ** 2)) / x.shape[0]
print("MSE: ",MSE)

plt.scatter(inputs['TotalFinishedArea'], outputs, color='blue')
plt.scatter(inputs['TotalFinishedArea'], test_y_hat, color='red')
plt.show()