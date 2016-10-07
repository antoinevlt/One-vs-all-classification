# One-vs-all-classification
Machine Learning Exercise 3: multi-class classification problem > one-vs-all-classification using regularized logistic regression in Octave/Matlab

I use logistic regression to recognize handwritten digits (from 0 to 9). I extend my previous implemention of logistic regression and apply it to one-vs-all classification.

Files included:
ex3.m - Octave/MATLAB script that steps you through part 1
ex3data1.mat - Training set of hand-written digits
displayData.m - Function to help visualize the dataset
fmincg.m - Function minimization routine (similar to fminunc)
sigmoid.m - Sigmoid function
lrCostFunction.m - Logistic regression cost function
oneVsAll.m - Train a one-vs-all multi-class classifier
predictOneVsAll.m - Predict using a one-vs-all multi-class classifier
