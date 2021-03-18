
The files in this folder provide a means for performing gradient descent on the housing data supplied.

This can be demonstrated with the following process:

1.

	Beginning with the preprocess_data.py, the data is imported and the two signifiagnt columns are selected for input and output.
	These columns then have rows culled where the Total Appraised Value was greater than two million, and where data did not exist for Total Finished Area.
	A function from the GD_Utils.py file then normalizes the whole dataset, with the option to use max-min or z-score normalization types.
	The rows of the dataset are shuffled.
	The percentages of training and validation data distribution are then set and the data is split, with test data being the remaining data not in the other two sets.
	These datasets are then written to .csv files in a folder.
	The two signifigant values from normalization are also saved to the folder in so that normalized values can be converted back to the range.

2. 
		
	
