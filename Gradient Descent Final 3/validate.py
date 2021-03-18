import GD_Utils as gdu
import numpy as np
import pandas as pd
import math
import pickle
import matplotlib.pyplot as plt

weights = pickle.load(open( "models/model1.p", "rb" ))

validation_data = pd.read_csv("datasets/validation_data.csv")

inputs = validation_data['TotalFinishedArea'].to_frame()
outputs = validation_data['TotalAppraisedValue']

inputs = gdu.add_polynomial(inputs,num=2)

inputs.insert(0,"ones",np.ones(inputs.shape[0]),True)
x = np.array(inputs)
y = np.array(outputs)

val_y_hat = x @ weights

MSE = (sum((val_y_hat - y) ** 2)) / x.shape[0]
print("MSE: ",MSE)

plt.scatter(inputs['TotalFinishedArea'], outputs, color='blue')
plt.scatter(inputs['TotalFinishedArea'], val_y_hat, color='red')
plt.show()