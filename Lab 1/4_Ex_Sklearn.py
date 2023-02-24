import numpy as np
import matplotlib as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
boston = datasets.load_boston()
X = boston.data
y = boston.target
feature_names = boston.feature_names
print("Feature names:", feature_names)
print("\nFirst 10 lines\n", X[:10])

# REMOVED FROM SIKIT LEARN
