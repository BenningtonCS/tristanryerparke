This files within this folder demonstrate Linear Regression using Gradient Descent on several different sets of input data.

The file GD_Utils.py contains several functions which provide methods for working with the data, and a major definition which performs the gradient descent.

Beginning with housing_data_prep.py the data is shuffled and split into three sets:
Training 80%
Validation 10%
Test 10%

Training:
	Then in the Train_Model.py file, the training data is imported and split into inputs and outputs.
	The script then normalizes the data to a 0-1 scale, saving the maxes and mins for later.

	This data is then sent to the Gradient Descent function and a row of 1s are added to the beginning of the input.
	An array of weights are initialized at zero.

	The cost function of this gradient is P(x) = sum((y^ - y))/2

	Then the GD loop begins by calculating y^ on line 25 with by multiplying the matrix x with the weight vector. 
	Then the errors are calculated by subtracting y^ - y. 

	The partial derivatives of P(x) with respect to each column of x are then calculated by iterating through each column of x and multiplying it with the errors, summed, and each is updated with these values. 

	The square of the magnitude of gradient is also calculated for use in the Backtracking Line Search and to possibly stop the loop if the error is small enough. 

	On line 33, an if statement detects if a fixed step size was passed to the function or no. If not it proceeds to use the BLS to calculate a step size. 

	The below lines converge on a step size for navigating the gradient descent(updating the weights).

	RSS is calculated.

	The GD loop ends, Output data is calculated and returned.


	Next the final weights and other signifigant information are saved to a file.

	A graph of the normalized final y_hat is shown over the normalized input data.


Validation:
	In Validate_Model.py, the validation data is imported.

	The saved model weights and original dataset bounds are imported.

	Validation data inputs are normalized using the maxes and mins from the training dataset.

	A column of ones is added to the beginning of the normalized validation input data.

	y_hat for the valiation data is calculated and then un-normalized with the bounds of the training outputs.
	(This is where I believe my issues is)

	Errors are calculated, MSE calculated.

	A graph is shown to visually inspect the data. 

Issue documented:
As I decrease the tolerance and increase the number of iterations for training, MSE for the training decreases.
The  MSE when using the validation set decreases but remains in the billions. 

Looking at both graphs, the y_hat for the validation set does not particularly look like a bad approximation, although there are some extreme outliers. 

I think my issue comes from confusion around normalization, or maybe just a typo/forgettful bad line of code.














