from sys import displayhook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Loading the dataset
data = pd.read_csv('Iris.csv')
plt.scatter(data['SepalLengthCm'], data['SepalWidthCm'])
plt.show()
# Show in form of histograms
plt.hist(data['SepalLengthCm'], bins=15)
plt.show()
# Bar chart
plt.bar(data['Species'], height=10, width=0.9, color='red')
# Count null values
print("Total Null Data:", data.isnull().sum())
# Displaying the samples
displayhook(data.loc[0: 4, 'SepalLengthCm'])
displayhook(data.loc[4:])
