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
	
	Sentance 1: "this","movie","sucked"]
	Vector 1: [1,1,1,0,0]
	Sentance 2: ["this","movie","was","awesome",awesome"]
	Vector 2: [0,1,1,2,1]

The first element of Vector 2 is zero as "sucked" did not appear in the sentance, following this pattern, element 2 corresponds to the single use of "this".
Element 4 is equal to two as the word "awesome" appeared twice in the review. 

This vector form allows us to describe each review as a vector with respect to the entire dataset. This vector then becomes the input for the training, and each review status as positive or negative is the output.

## Training ##

In train_model.py, the preprocessed data is imported and a bias term is added.

The weights are initalized and stochastic gradient ascent is started. 
With each major iteration, a y_hat is generated for each input x, error is calculated and the weights are updated.

The data is shuffled for the next major iteration.

Once the number of max iterations are reached, the program saves the weights to the local directory via pickle.

## Testing ##

In test_model.py, the preprocessed test data is imported, bias term is added, and variables to record true positves, true negatives, false positives and false negatives are initialized. 

A y_hat is then calculated by multiplying the test inputs with the weights from training. 

This y_hat is then iterated through, analyzed for error, and the type of error is classified as tp, tn, fp, fn. These values are counted across each example of y_hat.

Data about the accuracy of the model is printed.

## Conclusion ##

The following values were outputted when testing a model trained with 5 iterations. It seems to perform allright:
Notice that the y_hat values are not very close to zero or one.


	y_hat =  [0.89640848 0.99128675 0.64518875 0.53272699 0.89434714 0.82030305
	 0.86291456 0.25117742 0.91781794 0.94212182 0.85663408 0.95668143
	 0.84294248 0.7927724  0.74185889 0.88586365 0.67985367 0.69847764
	 0.89602562 0.69167853 0.81760211 0.00743559 0.22453635 0.4720574
	 0.09596233 0.07256152 0.27082278 0.18922279 0.61601285 0.19788354
	 0.46312747 0.21011164 0.1463821  0.04406928 0.57779708 0.52397415
	 0.50524972 0.04652873 0.62025142 0.06027629 0.09768938 0.26054497]
	tp =  20
	fp =  5
	fn =  1
	tn =  16
	accuracy =  0.8571428571428571
	precision =  0.8
	recall =  0.8571428571428571

This model was allowed 10 training iterations, it performs much better:

	y_hat =  [0.51452109 0.96022924 0.55406111 0.39581581 0.67999335 0.50155275
	 0.82215394 0.15751251 0.48503669 0.91601275 0.04557918 0.68684785
	 0.15859939 0.57016527 0.5538906  0.57578498 0.39325455 0.77603138
	 0.88210185 0.74241032 0.2365567  0.00008019 0.03863376 0.08032873
	 0.02815406 0.00886072 0.20406949 0.08787287 0.18520603 0.02118044
	 0.01597054 0.05888827 0.01573591 0.00717652 0.09611853 0.37693163
	 0.30655203 0.00389743 0.04584057 0.04208842 0.01142523 0.03836878]
	tp =  14
	fp =  0
	fn =  7
	tn =  21
	accuracy =  0.8333333333333334
	precision =  1.0
	recall =  0.8333333333333334
This model was allowed 20 training iterations, it performs extremely well:
These y_hat values seem to be much more "confident" than the other two models, preidcting a serious positive or negative for each review.

	y_hat =  [0.94513725 0.99756862 0.861876   0.49246622 0.92646583 0.81221372
 	 0.93531945 0.17571723 0.58204478 0.98806863 0.74714729 0.99733376
 	 0.42594287 0.91676947 0.64930849 0.95472875 0.70288807 0.91869822
 	 0.91904822 0.91554061 0.7237399  0.00732241 0.09905718 0.07734968
 	 0.04199171 0.01748766 0.21267941 0.07769827 0.50164798 0.03008378
 	 0.02618186 0.11604311 0.08135671 0.02816399 0.17705487 0.39084311
 	 0.37061482 0.0182972  0.29607181 0.04488934 0.0533805  0.12530293]
	tp =  18
	fp =  1
	fn =  3
	tn =  20
	accuracy =  0.9047619047619048
	precision =  0.9473684210526315
	recall =  0.9047619047619048 
In the beginning I began this project simply, trying to create a preprocessing script which numbered how many positive or negative words were in eac anyh review. I found that most of the preprocessing methods I tried failed to single out words which were signifigantly attached to positive or negative. Therefore the bag of words approach using vectors to describe each review seemed much more promising, and required a less complicated preprocessing script, but more processing power.

Through some optimization tips and debugging with justin, I was able to seriously speed up my bag of words code so that it calculated in seconds.
Overall, the training seemed to be sucessful and I belive the model would work on reviews which have signifigant negative or positive words. Neutral ones it may classify wrongly. A larger dataset might be in hand for this to get better, but the bag of words approach really helps with the neutral reviews as it can take into account the entire sentance and make judgment based on that.



