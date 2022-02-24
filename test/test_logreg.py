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

	"""
	X_train, X_test, y_train, y_test = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Diastolic Blood Pressure', 'Body Mass Index','Computed tomography of chest and abdomen',
       'Plain chest X-ray (procedure)','Body Weight', 'Body Height','Systolic Blood Pressure', 
	'Low Density Lipoprotein Cholesterol', 'High Density Lipoprotein Cholesterol', 
	'Triglycerides','Total Cholesterol'], split_percent=0.8)
	lr = regression.LogisticRegression(X_train.shape[1])
	lr.train_model(X_train, y_train, X_test, y_test)

	# Check that gradient is being calculated correctly

	# Check that loss function is correct and that 
	# there is reasonable losses at the end of training
	assert np.all(np.array(lr.loss_history_train[-5:]) < 0.70)
	assert np.all(np.array(lr.loss_history_val[-5:]) < 0.70)



def test_predict():
	"""
	
	"""
	X_train, X_test, y_train, y_test = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Diastolic Blood Pressure', 'Body Mass Index',
	'Body Weight', 'Body Height','Systolic Blood Pressure', 
	'Low Density Lipoprotein Cholesterol', 'High Density Lipoprotein Cholesterol', 
	'Triglycerides','Total Cholesterol'], split_percent=0.8)
	lr = regression.LogisticRegression(X_train.shape[1])
	lr.train_model(X_train, y_train, X_test, y_test)
	
	# Check that self.W is being updated as expected
	# and produces reasonable estimates for NSCLC classification

	# Check accuracy of model after training

test_updates()
test_predict()