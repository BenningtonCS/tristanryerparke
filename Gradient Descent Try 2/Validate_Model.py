import GD_Utils as gdu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
np.set_printoptions(suppress=True)

#Import the val data
val_data = pd.read_csv("datasets/validation_data.csv")
#print(housing_data)

#Parse the data into inputs and outputs
raw_val_inputs = val_data["TotalFinishedArea"].to_frame()
raw_val_outputs = val_data["TotalAppraisedValue"].to_frame()

#Add polynomials
raw_val_inputs = gdu.add_polynomials(raw_val_inputs,num=2)

#Load the model
model = pickle.load(open( "model1.p", "rb" ))
training_input_bounds = model["input_bounds"]
training_output_bounds = model["output_bounds"]
weights = model["weights"]

#Normalize with training bounds
print("Normalizing inputs")
val_inputs = gdu.normalize(raw_val_inputs,input_bounds=training_input_bounds)[0]

val_inputs.insert(0,"ones",np.ones(val_inputs.shape[0]),True)

x = np.array(val_inputs)

val_y_hat = x @ weights

val_y_hat = gdu.un_normalize(val_y_hat,training_output_bounds)
#The bounds of these do not seem right.

val_y = np.array(raw_val_outputs.iloc[:, 0])

errors = val_y_hat - val_y

MSE = sum(errors ** 2) / x.shape[0]

print("MSE: ",MSE)
#Issue
#The MSE values seem way to high but decrease when the model is allowed more iterations and smaller tolerance.
#Meaning that I am not overfitting but something is off (most likley with the un-normalization).






