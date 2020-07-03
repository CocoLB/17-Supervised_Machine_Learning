# 17-Supervised_Machine_Learning
Module 17: Supervised Machine Learning and Credit Risk

## Challenge Overview
We've been tasked with building and evaluating different machine learning models to asses credit risk.

## Resources
- Python 3.7, Numpy 1.11+, Scipy 0.17+, Scikit-learn 0.21+
- LoanStats_2019Q1.csv (data from LendingClub) (zipped in Github)

## Challenge Summary
In both Notebooks we follow the same steps to preprocess the data:
   
   - we split the data: y being the target "loan_status" column, and X all the other columns
   - from X, we find all the columns with an "object" (ie "string") format
   - we encode these columns with the get_dummies encoder
   - we create the train and test sets
   
For each model, we then follow the same steps:
   
   - we resample the training data with the Sampler/Classifier model
   - we train the Logistic Regression Model using that resampled data
   - we apply this model on the testing data
   - we calculate the balanced accuracy score, and display the condusion matrix, and imbalanced cl;assification report.
   
   
The scores we are looking at are 
   - balanced accuracy score to show the genral accuracy of the model
   - precision - all the scores are low, which means that a wrongly predicted "low risk" will be corected afterwards
   - recall - we want to find the highest possible, to have the fewest "wrongly predicted high risks"
   
Credit_risk_resampling Notebook: comparing the first 4 models, the Naive Random Oversampling model is the best one with the highest Recall and Accuracy score, but the Recall is not as high as we could hope, at 0.69.

Credit_risk_ensemble Notebook: 
The Balance Random Forest Classifier has higher accuracy score and precision, but not a higher recall, contrary to the AdaBoost Classifier model whose metrics are all far higher, with a recall of 0.92 (and an accuracy of 0.93).

### Conclusion: The AdaBoost Classifier model is, by far, the best model, and the only one that is recommended.


