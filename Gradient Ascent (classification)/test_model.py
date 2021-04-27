import numpy as np
import ascent_utils as au
import pickle
np.set_printoptions(suppress=True)

x, y = pickle.load(open("modelData/test_data.p", "rb" ))
x = np.hstack((np.ones((x.shape[0],1)),x))
w = pickle.load(open("modelData/weights.p", "rb"))
tp, tn, fp, fn = 0, 0, 0, 0

y_hat = au.sigmoid(np.array([(-1 * w.T) @ xi for xi in x]))

for y_hat_item, y_item in zip(y_hat,y):
    if y_hat_item > 0.5:
        if y_item == 1:
            tp += 1
        else:
            fp += 1
    else:
        if y_item == 1:
            fn += 1
        else:
            tn += 1

print("y_hat = ",y_hat)

print("tp = ",tp)
print("fp = ",fp)
print("fn = ",fn)
print("tn = ",tn)

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print("accuracy = ",accuracy)
print("precision = ",precision)
print("recall = ",accuracy)