![BuildStatus](https://github.com/apblair/project6/actions/workflows/main.yml/badge.svg?event=push)
# Project 6
Logistic regression and application to medical record data

# Assignment

## Overview
In class, we described how to derive an OLS estimators in a linear regression model, which can be used to identify a best fit line. For this project, you will be implementing a logistic regression model using the same framework. Logistic regression is useful for classification because the function outputs a value between 0 and 1, which corresponds to categorical classification. 
https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e  

For this project, you will be given a set of (simulated) medical record data from patients with small cell and non-small cell lung cancers (1). Code used to generate the data can be found in the patient.ipynb. The goal of this project is to write a class that performs logistic regression and apply it to this dataset to predict whether a patient has small cell or non-small cell lung cancer, using features that draw from demographic, medication, lab values, and/or procedures performed prior to their diagnosis.  

1) Jason Walonoski, et al. Synthea: An approach, method, and software mechanism for generating synthetic patients and the synthetic electronic health care record, Journal of the American Medical Informatics Association, Volume 25, Issue 3, March 2018, Pages 230–238, https://doi.org/10.1093/jamia/ocx079

## Dataset 
Class labels are encoded in the "NSCLC" column of the dataset, with 1=NSCLC and 0=small cell. A set of features has been pre-selected for you to use in your model during testing, but you are encouraged to submit unit tests that look at different features. The full list of features can be found in the utils.py file.  

As a side note, for people interested in medical informatics, we're incredibly lucky at UCSF to have access to de-identified patient data which are often much more detailed/standardized than the dataset provided here and have far more patients (>>1.5M patients annually, not to mention the UC-wide data). UCSF also has de-identified clinical notes for NLP research, and they are now all available without an IRB! 

## Logistic regression
To allow for binary classification using logistic regression, we used a sigmoid function to model the data. Just like in linear regression, we will define a loss function to keep track of how well the model performs, but instead of mean-squared error, you will need to implement a log loss (binary cross-entropy) function. This function minimizes the error when the predicted y is close to an expected value of 1 or 0. Here are some resources to get you started: 
* https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
* https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11
* https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

## Project requirements (10 points total)

[ TODO ] Complete the Logreg class with your implementation of the algorithm (5 points)
  * complete the 'calculate_gradient' method
  * complete the 'loss_function' method
  * complete the 'make_prediction' method

  * effective API structure with good documentation and commenting

[ TODO ] Unit Testing (3 points)
  * check that fit appropriately trains model & weights get updated
  * check that loss approaches 0
  * check that predict is working 

[ TODO ] Packaging (2 point)
  * pip installable module
  * github actions (install + pytest)


# Getting Started
To get started you will need to fork this repo onto your own github account. Work on your codebase from your own repo and commit changes. I have listed the minimum python module requirements in `requirements.txt` 

# Additional notes
Try tuning the hyperparameters if you find that your model doesn't converge. Hint: too high of a learning rate or too large of a batch size can sometimes cause the model to be unstable (loss function goes to infinity).

Sklearn has some sample datasets (diabetes, breast cancer) that you can also use for testing. 

