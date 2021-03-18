
The files in this folder provide a means for performing gradient descent on the housing data supplied.

This can be demonstrated with the following process:

1. Pre-Processing

	
		Beginning with the preprocess_data.py: 
		The data is imported and the two signifiagnt columns are selected for input and output.
		These columns then have rows culled where the Total Appraised Value was greater than two million, 
		and where data did not exist for Total Finished Area.
		A function from the GD_Utils.py file then normalizes the whole dataset, 
		with the option to use max-min or z-score normalization types.
		The rows of the dataset are shuffled.
		The percentages of training and validation data distribution are then set and the data is split, 
		with test data being the remaining data not in the other two sets.
		These datasets are then written to .csv files in a folder.
		The two signifigant values from normalization are also saved to the folder
		so that normalized values can be converted back to the range

2. Training

		In train.py, the training dataset is imported and split into inputs and outputs. 
		In the example script provided, an extra row of the data squared is added, 
		the option to add additional columns with x^3, x^4 is avaliable by changing the num=2 parameter. 
		A column of 1's are also added to the beginning.
	
		The training begins using the gradient descent function from GD_Utils.py, 
		which starts by converting the inputs and outputs into numpy arrays.
		Additional arrays are created for weights and partial derivatives of the cost function.
		The loop begins:
		y_hat is calculated by multiplying the inputs with the weights, 
		and errors calculated by subtracting y from y_hat.
		The partial derivatives of the cost function are updated within a for loop below.
		The magnitude of the gradient ** 2 is also calculated for later use.
	
		If no given step size was passed to the function, 
		the Backtracking Line Search will automaticlly be used. Picking a step size at each 
		iteration which is used to update the weights and move down the gradient. 
	
		RSS is calculated and the number of iterations are updated.
		A loop updates the values of the weights.
		Conditions to break the loop are checked.
	
		After the loop is over, the weights, MSE and final y_hat are 
		printed and passed back to the training script.

		Weights are saved to a local file to be referenced later.
		A graph of the normalized data versus the normalized y_hat is 
		shown to view the visual accuracy of the model.
	
		With tolerance 0.001, x, x^2 and max iterations = 1000, these weights and mse were produced:
		weights:  [ 0.03471246  2.0610451  -1.32873416]
		MSE: 0.0026409661119723613
		A 5000 iteration model with x, x^2 and x^3 scored a little better but not by much:
		[ 0.02483094  2.64144582 -1.57892856 -0.82775003]
		MSE: 0.0023699797677217047
	
	
3. Validation
	
		In validation.py, the weights from training are imported as well as the validation dataset.
		The validation set is processed to match the training set data in 
		its final form before the gradient descent was performed.
		This data is then multiplied by the weights and a y_hat for the validation data is calculated. 

		The MSE is calculated and printed.
	
		A graph is shown with the normalized data versus the prediction of the y values deemed by the imported weights.
		This allows us to see that the model is similarly accurate on the validation set.
		Scoring:
		MSE:  0.003738713477811529
		This is a little more error than the training set, but considering the number of outlying datapoints in the set, 
		I think this difference is negligible.
		The 5000 x, x^2 x^3 iteration model:
		MSE:  0.003522480007606554
	
4. Testing
	
		In test.py the exact same validation process is performed with the test data.
		The 1000 iteration model scored much better than on the validation dataset.
		
		MSE:  0.0029640192111095687
		The 5000 x, x^2 x^3 iteration model:
		MSE:  0.0026016822146469553
		Seems to perform a little better.
	
	
5. Additional script
	
		An additional script predict_value_model1.py will import the weights from a training the model with train.py, 
		and use them to predict a total apprasial value from a user inputted total finshed area. 
		It does this by normalizing the value, applying it to the weights 
		in a similar method to the validation and training sets. 
		It then re-maps the output value to the scale of appraised values 
		based on the entire dataset and prints it in dollar format.
	
	
	
	 
	
	
	
	
	 
	
