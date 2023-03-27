import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Loading th dataset
data_Fetch = pd.read_csv('Exercise-CarData.csv')
print("\nData :\n", data_Fetch)
print("\nData statistics\n", data_Fetch.describe())
# All rows, all columns except last
X = data_Fetch.iloc[:, :-1].values
# Only last column
Y = data_Fetch.iloc[:, -1].values
print("\n\nInput : \n", X)
print("\n\nOutput: \n", Y)
# removing row with all null value
data_Fetch.dropna(axis=1, how='all', inplace=True)
print("\nNew Data :", data_Fetch)
# Removing the row with any one null values
data_Fetch.dropna(axis=0, how='any', inplace=True)
print("\nNew Data :", data_Fetch)
data_Fetch = data_Fetch.replace(to_replace="??", value=50000)

data_Fetch = data_Fetch.replace(to_replace="three", value=3)
print("\nNew Data :", data_Fetch)
print("*********************************************DATA TRANSSFORMATION*********************************************")
# All rows, all columns except last
X = data_Fetch.iloc[:, :-1].values
# Only last column
Y = data_Fetch.iloc[:, -1].values
print("\n\nInput : \n", X)
print("\n\nOutput: \n", Y)
X_new = data_Fetch.iloc[:, 1:4].values
print("\n\nX for transformation : \n", X_new)
#
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_new)
print("\n\nScaled X : \n", X_scaled)
#
std = StandardScaler()
X_std = std.fit_transform(X_new)
print("\n\nStandardized X : \n", X_std)
print("*********************************************One Hot & Labl Encoding*********************************************")
print("\nData :\n", data_Fetch)
print("\nData statistics\n", data_Fetch.describe())
# All rows, all columns except last
X = data_Fetch.iloc[:, :-1].values
# Only last column
Y = data_Fetch.iloc[:, -1].values
print("\n\nInput : \n", X)
print("\n\nOutput: \n", Y)
le = LabelEncoder()
X[:, 4] = le.fit_transform(X[:, 4])
print("\n\nInput : \n", X)
print(X[:, 4])
dummy = pd.get_dummies(data_Fetch['FuelType'])
print("\n\nDummy :\n", dummy)
data_Fetch = data_Fetch.drop(['FuelType'], axis=1)
data_Fetch = pd.concat([dummy, data_Fetch], axis=1)
print("\n\nFinal Data :\n", data_Fetch)
onehotencoder = OneHotEncoder()
data_Fetch = pd.read_csv('/content/drive/MyDrive/exercise-car-data.csv')
# reshape the 1-D country array to 2-D as fit_transform expects 2-D and finally fit the object
x = onehotencoder.fit_transform(
    data_Fetch.FuelType.values.reshape(-1, 1)).toarray()
print(x)
dfOneHot = pd.DataFrame(x, columns=["FuelType_"+str(int(i)) for i in range(4)])
df = pd.concat([data_Fetch, dfOneHot], axis=1)  # column
df = df.drop(['FuelType'], axis=1)
print(df.head())
