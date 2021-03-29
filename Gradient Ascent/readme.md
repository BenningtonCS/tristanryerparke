 # Binary Classification with Gradient Ascent #

Goal: given seperate datasets of positive and negative movie reviews, train and test a binary classifiaction model using gradient ascent which can classify a given movie review as positive or negative.

## Preprocessing ##

The given data needs to be converted into numerical (numpy array) format that the gradient ascent training script can use to create the weights.

preprocess.py does this for both training and test sets, beginning by splitting the data into individual reviews, then turning these into lists of words.
A dictionary is then created with the keys being a word and values being an integer which signifies that word. The length of the dictionary is the number of unique words in the given dataset. 
The sentances are then converted into lists of integers, with the dictionary being the link between the two as such. 
Dict: {"sucked":1,"this":2,"movie":3,"awsome":4",was":5}
Sentance: ["this","movie","sucked"]
Integer Sentance: [2,3,1]
These lists of integers allow creation of a dictionary-length vector where each element represents the use of a word.
Sentance: ["this","movie","was","awesome",awesome"]
Vector [0,1,1,2,1]
The first element is zero as "sucked" did not appear in the sentance, following this pattern, element 2 corresponds to the single use of "this".
Element 4 is equal to two as the word "awesome" appeared twice in the review. 

This vector form allows us to describe each review as a vector with respect to the entire dataset. This vector then becomes the input for the training, and each review status as positive or negative is the output.

