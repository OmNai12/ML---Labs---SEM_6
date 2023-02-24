# importing libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
# Mounting the dataset
dataset = pd.read_csv('Dataset2.csv')
print("Data :- \n", dataset)
# Data stats
print("Data Statistics :- \n", dataset.describe())
# creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
outlook_encoded = le.fit_transform(dataset['Outlook'])
print("Outlook:", outlook_encoded)
temp_encoded = le.fit_transform(dataset['Temp'])
print("Temp:", temp_encoded)
humidity_encoded = le.fit_transform(dataset['Humidity'])
print("Humidity:", humidity_encoded)
wind_encoded = le.fit_transform(dataset['Wind'])
print("Wind:", wind_encoded)
play_encoded = le.fit_transform(dataset['Class'])
print("Play:", play_encoded)
# Combinig Outlook,Temp,Humidity and Wind into single listof tuples
features = list(zip(outlook_encoded, temp_encoded,
                    humidity_encoded, wind_encoded))
print("Features:", features)
# Create a Classifier
model = MultinomialNB()
train_data, test_data, y_train, y_test = train_test_split(
    features, play_encoded, test_size=0.5)
# Train the model using the training sets
model.fit(features, play_encoded)
y_pred = model.predict(test_data)
print("The precision is : ", precision_score(y_test, y_pred))
print("The Accuracy is  : ", accuracy_score(y_test,  y_pred))
print("The Recall is    : ", recall_score(y_test,  y_pred))
