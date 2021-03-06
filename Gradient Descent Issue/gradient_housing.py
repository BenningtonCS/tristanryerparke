import tristan_ML as tml
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

data = pd.read_csv("Machine Learning S21/test_data.csv")

#When using real data an error comes up:
x = data['TotalFinishedArea'].to_frame()
y = data['TotalAppraisedValue'].to_numpy()

#When using some simple data with the same format there are no issues:
#x = pd.DataFrame({"something":[0,  1,  2,  3,  4]})
#y = np.array([1,  2,  5,  10,  17])

x = np.array(tml.add_exponents(x,num=1))

print(x)
print(y)

step_size = 0.005
tolerance = 0.01
max_iterations = 10000

tml.gradient_descent(x,y,step_size,tolerance,max_iterations)