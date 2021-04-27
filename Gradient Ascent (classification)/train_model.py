import numpy as np
import random
import ascent_utils as au
import pickle

x, y = pickle.load(open( "modelData/training_data.p", "rb" ))

step_size = 5e-2
mag_sq = float('inf')

x = np.hstack((np.ones((x.shape[0],1)),x))
w = np.zeros(x.shape[1])
derivatives = np.zeros(x.shape[1])
y_hat = np.zeros(y.shape[0])
index_list = list(range(0,x.shape[0]))

iterations = 0
max_iterations = 10

while True:

    for i in index_list:
        y_hat = au.sigmoid((-1 * w.T) @ x[i])
        error = y[i] - y_hat 
        derivative = x[i] * error
        w = w + derivative * step_size
    
    mag_sq = np.dot(derivative,derivative)

    print("iterations = ",iterations)
    print("mag_sq = ",mag_sq)
    print("w: ",w)

    iterations += 1

    random.shuffle(index_list)

    if iterations >= max_iterations:
        break

pickle.dump(w, open("modelData/weights.p", 'wb'))
print("Saved weights to file.")