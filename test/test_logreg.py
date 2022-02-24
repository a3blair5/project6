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
	X_train, X_test, y_train, y_test = utils.loadDataset(split_percent=0.8)
	lr = regression.LogisticRegression(X_train.shape[1])
	lr.train_model(X_train, y_train, X_test, y_test)

	# Check that gradient is being calculated correctly
	
	
	# Check that loss function is correct and that 
	# there is reasonable losses at the end of training
	assert all([True for loss in lr.loss_history_train if loss < 1 or loss > 0]) 

def test_predict():
	"""
	
	"""
	X_train, X_test, y_train, y_test = utils.loadDataset(split_percent=0.8)
	lr = regression.LogisticRegression(X_train.shape[1])
	lr.train_model(X_train, y_train, X_test, y_test)
	
	# Check that self.W is being updated as expected
	# and produces reasonable estimates for NSCLC classification

	# Check accuracy of model after training

test_updates()
# test_predict()