import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(r"C:\Users\rapol\Downloads\DATA SCIENCE\4. Dec\1st,2nd - logistic, pca\2.LOGISTIC REGRESSION CODE\logit classification.csv")

# Features and target
x = data.iloc[:, [2, 3]].values
y = data.iloc[:, -1].values

# Split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


# Train the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# Predict
y_pred = classifier.predict(x_test)
print(y_pred)

# confusion_matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

#accuracy
from sklearn.metrics import accuracy_score
cm=accuracy_score(y_test, y_pred)
print(cm)

bias=classifier.score(x_train,y_train)
bias

veriance=classifier.score(x_test,y_test)
veriance