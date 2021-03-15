import GD_Utils as gdu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)#

#Import the training data
training_data = pd.read_csv("datasets/training_data.csv")
#print(housing_data)

#Parse the data into inputs and outputs
training_inputs = training_data["TotalFinishedArea"].to_frame()
training_outputs = training_data["TotalAppraisedValue"].to_frame()

#Add polynomials
training_inputs = gdu.add_polynomials(training_inputs,num=2)

#Normalize
print("Normalizing inputs")
training_inputs, input_maxes, input_mins = gdu.normalize(training_inputs,n_type="max/min") 
print("Normalizing outputs")
training_outputs, output_max, output_min = gdu.normalize(training_outputs,n_type="max/min") 

#Train model
results = gdu.gradient_descent(training_inputs, training_outputs, tolerance=0.001, max_iterations=5000, debug=True)
np.save("weights.npy",results["weights"])

final_weights = np.load("weights.npy")

#Show normalized results
#plt.scatter(training_inputs["TotalFinishedArea"],training_outputs["TotalAppraisedValue"])
#plt.scatter(training_inputs["TotalFinishedArea"],results["y_hat"])
#plt.show()

print(final_weights)


#Validate model
validation_data = pd.read_csv("datasets/training_data.csv")

validation_inputs = validation_data["TotalFinishedArea"].to_frame()
validation_outputs = validation_data["TotalAppraisedValue"].to_frame()

validation_inputs = gdu.add_polynomials(validation_inputs,num=2)


















