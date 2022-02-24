"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
import regression
from regression import *

def test_updates():
	"""
	Unit test for gradient descent and loss function
	"""

	# Check that gradient is being calculated correctly
	lr_gradient_check = LogisticRegression(num_feats = 6, max_iter = 10, learning_rate = 0.00001, batch_size = 12)
	X,y = np.array([[1, 2, 3], [1, 2, 3]]), np.array([0,0])
	lr_gradient_check.W = np.array([3,3,3]) 
	gradient_check = lr_gradient_check.calculate_gradient(X, y)
	assert np.allclose(gradient_check, np.array([1, 2, 3]))
	
	X_train, X_test, y_train, y_test = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Diastolic Blood Pressure', 'Body Mass Index','Computed tomography of chest and abdomen',
       'Plain chest X-ray (procedure)','Body Weight', 'Body Height','Systolic Blood Pressure', 
	'Low Density Lipoprotein Cholesterol', 'High Density Lipoprotein Cholesterol', 
	'Triglycerides','Total Cholesterol'], split_percent=0.8)
	lr = regression.LogisticRegression(X_train.shape[1])
	lr.train_model(X_train, y_train, X_test, y_test)

	# Check that loss function is correct and that 
	# there is reasonable losses at the end of training
	assert np.all(np.array(lr.loss_history_train[-5:]) < 0.70)
	assert np.all(np.array(lr.loss_history_val[-5:]) < 0.70)

	# Check if loss is lower with more iterations
	lr_5_iters,lr_500_iters = regression.LogisticRegression(X_train.shape[1], max_iter=5),regression.LogisticRegression(X_train.shape[1], max_iter=500)
	lr_5_iters.train_model(X_train, y_train, X_test, y_test)
	lr_500_iters.train_model(X_train, y_train, X_test, y_test)
	assert lr_500_iters.loss_history_train[-1] < lr_5_iters.loss_history_train[-1]

def test_predict():
	"""
	Unit test to check model weights, NSCLC estimates, and accuracy of the model after training
	"""
	
	# Check that self.W is being updated as expected
	X_train, X_test, y_train, y_test = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Diastolic Blood Pressure', 'Body Mass Index','Computed tomography of chest and abdomen',
       'Plain chest X-ray (procedure)','Body Weight', 'Body Height','Systolic Blood Pressure', 
	'Low Density Lipoprotein Cholesterol', 'High Density Lipoprotein Cholesterol', 
	'Triglycerides','Total Cholesterol'], split_percent=0.8)
	lr_weight_check = regression.LogisticRegression(X_train.shape[1], max_iter=2000, batch_size=12)
	initial_weights = lr_weight_check.W
	lr_weight_check.train_model(X_train, y_train, X_test, y_test)
	assert np.allclose(initial_weights, lr_weight_check.W) ==False

	# Check model produces reasonable estimates for NSCLC classification
	# Check if predictions are between 0 and 1
	y_predicted = lr_weight_check.make_prediction(X_test)
	i, j = 0, 1
	for predicted in y_predicted:
		if predicted < i or predicted >= j :
			raise ValueError("Error: prediction not in predicted range [0,1].")
	assert np.round(np.min(y_predicted),1) == 0, "Assessment Check: The model is not producting reasonable estimates for NSCLC classification"
	assert np.round(np.max(y_predicted), 1) == 1, "Assessment Check: The model is not producing reasonable estimates for NSCLC classification." 

	# Check accuracy of model after training
	y_predicted_labels = [0 if sample < 0.5 else 1 for sample in y_predicted]
	accuracy = 100 * np.sum(y_test == y_predicted_labels) / len(y_test)
	assert accuracy > 60, "Assessent Check: The model should produce a score higher than a coin flip. The cuttoff is set to 60%."
