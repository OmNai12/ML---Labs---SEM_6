# By :- Om Nai
# Roll Number :- CE072

from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# Loading data set
df = pd.read_csv("data.csv")
print(df.head())
x = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
# One hot encoding
print(x)
print(y)
# print(df.describe())
# Train test
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=1)
# Decision Tree
clf = DecisionTreeClassifier(
    criterion="entropy", random_state=38, max_leaf_nodes=38)
# Train Decision Tree Classifer
clf = clf.fit(xtrain, ytrain)
# Predict the response for test dataset
y_pred = clf.predict(xtest)
print(y_pred)
print("Accuracy:", metrics.accuracy_score(ytest, y_pred))
precision = precision_score(ytest, y_pred, average=None)
recall = recall_score(ytest, y_pred, average=None)
print('Accuracy  : ', accuracy_score)
print('Precision : '.format(precision))
print('Recall    : '.format(recall))
