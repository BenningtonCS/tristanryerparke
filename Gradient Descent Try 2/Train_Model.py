import GD_Utils as gdu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
np.set_printoptions(suppress=True)

#Import the training data
training_data = pd.read_csv("datasets/training_data.csv")
#print(housing_data)

#Parse the data into inputs and outputs
raw_training_inputs = training_data["TotalFinishedArea"].to_frame()
raw_training_outputs = training_data["TotalAppraisedValue"].to_frame()

#Add polynomials
raw_training_inputs = gdu.add_polynomials(raw_training_inputs,num=2)

#Normalize
print("Normalizing inputs")
training_inputs, input_bounds = gdu.normalize(raw_training_inputs) 
print("Normalizing outputs")
training_outputs, output_bounds = gdu.normalize(raw_training_outputs) 

#Train model
results = gdu.gradient_descent(training_inputs, training_outputs, tolerance=0.001, max_iterations=5000, debug=True)

#Save a file containing the model and the bounds of its inputs
save_model_dict = {"weights":results["weights"],
             "input_bounds":input_bounds,
             "output_bounds":output_bounds}

pickle.dump(save_model_dict,open( "model1.p", "wb" ))

#Show normalized results
plt.scatter(training_inputs["TotalFinishedArea"],training_outputs["TotalAppraisedValue"])
plt.scatter(training_inputs["TotalFinishedArea"],results["y_hat"])
plt.show()
#looking at this normalized graph, the values seem good.


















