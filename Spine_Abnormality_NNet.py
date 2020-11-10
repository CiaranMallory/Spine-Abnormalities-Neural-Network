import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.neural_network import MLPClassifier 
import pandas as pd

# Data Processing
df = pd.read_csv('Dataset_spine.csv')
df.replace('?', -99999, inplace=True)
df = df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Class_att']]
# Replacing Categorical data with Numeric values
df = df.replace(to_replace='Abnormal', value=1)
df = df.replace(to_replace='Normal', value=2)
#print(df.head)

X = np.array(df.drop(['Class_att'], 1))
y = np.array(df['Class_att'])

# Model Training
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=1)

clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)

# Example sets to test model
set1 = [60,22,35,44,97,-0.12,0.7,11,13.8,14.3]
set2 = [33,12,23,43,56,-0.6,0.4,2,34,12]

example_measures = np.array(set2)
example_measures = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)
#print(prediction)

if (prediction == 1):
	print("Result: Abnormal Spine")

elif (prediction == 2):
	print("Result: Normal")