import pickle
import ascent_utils as au

#Training Data:

training_data = au.import_data("movieData/pos_sample.txt", "movieData/neg_sample.txt")
training_vocab = au.create_vocab(training_data, min_uses=2)
training_intWords = au.to_intWords(training_data, training_vocab)
training_x, training_y = au.create_arrays(training_vocab, training_intWords, training_data)

check_index = 0
print("Sentance: ", training_data[check_index][0])
print("Intword: ", training_intWords[check_index])
print("x: ", training_x[check_index])
print("y: ", training_y[check_index])

pickle.dump((training_x, training_y), open("modelData/training_data.p", 'wb'))
print("Saved training data.")

#Test Data:
test_data = au.import_data("movieData/pos_test.txt", "movieData/neg_test.txt")
test_intWords = au.to_intWords(test_data, training_vocab)
test_x, test_y = au.create_arrays(training_vocab, test_intWords, test_data)

print("Test x:",test_x)
print("Test y:",test_y)

pickle.dump((test_x, test_y), open("modelData/test_data.p", 'wb'))
print("Saved test data.")




