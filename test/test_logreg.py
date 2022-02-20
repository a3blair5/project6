"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
from regression import *

def test_updates():
	"""

	"""
	# Check that your gradient is being calculated correctly
	
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training

	pass

def test_predict():
	# Check that self.W is being updated as expected
	# and produces reasonable estimates for NSCLC classification

	# Check accuracy of model after training

	pass

def test_dev():
	X_train, X_test, y_train, y_test = utils.loadDataset(split_percent=0.8)
	print(X_train.shape)
	print(y_train.shape)

test_dev()