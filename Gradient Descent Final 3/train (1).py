import GD_Utils as gdu
import numpy as np
import pandas as pd
import math
import pickle
import matplotlib.pyplot as plt

training_data = pd.read_csv("datasets/training_data.csv")

inputs = training_data['TotalFinishedArea'].to_frame()
outputs = training_data['TotalAppraisedValue']

inputs = gdu.add_polynomial(inputs,num=3)

model = gdu.gradient_descent(inputs, outputs, tolerance=0.001, max_iterations=5000)

weights = model["weights"]

pickle.dump(weights,open( "models/model1.p", "wb" ))

plt.scatter(inputs['TotalFinishedArea'], outputs, color='blue')
plt.scatter(inputs['TotalFinishedArea'], model['y_hat'], color='red')
plt.show()
