from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Dataset
Outlook = ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny',
           'Overcast',
           'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Overcast', 'Overcast',
           'Sunny']
Temperature = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
               'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
Humidity = ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
            'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High']
Wind = ['False', 'True', 'False', 'False', 'False', 'True', 'True',
        'False', 'False', 'False', 'True', 'True', 'False', 'True']
# Class Label:
Play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
        'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
# Label Encoding
le = preprocessing.LabelEncoder()
#
Outlook_encoded = le.fit_transform(Outlook)
Outlook_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Outllok mapping:", Outlook_name_mapping)
#
Temperature_encoded = le.fit_transform(Temperature)
Temperature_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Temperature mapping:", Temperature_name_mapping)
#
Humidity_encoded = le.fit_transform(Humidity)
Humidity_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Humidity mapping:", Humidity_name_mapping)
#
Wind_encoded = le.fit_transform(Wind)
Wind_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Wind mapping:", Wind_name_mapping)
#
Play_encoded = le.fit_transform(Play)
Play_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Play mapping:", Play_name_mapping)
print("\n\n")
print("Weather:", Outlook_encoded)
print("Temerature:", Temperature_encoded)
print("Humidity:", Humidity_encoded)
print("Wind:", Wind_encoded)
print("Play:", Play_encoded)
# Zip the features
features = list(zip(Outlook_encoded, Temperature_encoded,
                Humidity_encoded, Wind_encoded))
print("Features:", features)
# Test and train
X_train, X_test, y_train, y_test = train_test_split(
    features, Play_encoded, test_size=0.5)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
# Cost complexcity
clf = DecisionTreeClassifier(criterion="entropy", random_state=100)
# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = clf.predict(X_test)
print(y_pred)
# Metrixs
print(confusion_matrix(y_test, y_pred))
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Accuracy  : {}'.format(accuracy_score))
print('Precision : {}'.format(precision))
print('Recall    : {}'.format(recall))
