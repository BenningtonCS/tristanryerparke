import GD_Utils as gdu
import pickle
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)

#This script only works if max/min normalization is used when splitting up the data and for training the model.

bounds = pickle.load(open( "datasets/bounds.p", "rb" ))
weights = pickle.load(open( "models/model1.p", "rb" ))

usr_val = float(input("Please type the finished area of a home and press enter... "))
usr_val = (usr_val - bounds[0][0]) / (bounds[0][1] - bounds[0][0])
usr_frame = pd.DataFrame(data=[usr_val],columns=['usr_val'])

usr_input = gdu.add_polynomial(usr_frame,num=3)
usr_input.insert(0,"ones",np.ones(usr_input.shape[0]),True)
x = np.array(usr_input)

usr_y_hat = x @ weights

usr_raw_answer = (usr_y_hat * (bounds[0][1] - bounds[0][0])) + bounds[0][0]

print("Model 1 calculates your apprasial at ${:,.2f}".format(usr_raw_answer[0]))









