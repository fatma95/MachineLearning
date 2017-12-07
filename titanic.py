import numpy as np
import pandas as pd
import csv
from pandas import *
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Training Data
train_data = pd.read_csv("train_titanic.csv")
features = ["Pclass","Sex","Age"] #feature selection
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())
train_data["Sex"] = train_data["Sex"].apply(lambda sex:1 if sex=="male" else 0)
survived = train_data["Survived"].values
train_data = train_data[features].values


#Test data
test_data = pd.read_csv("test_titanic.csv")
features = ["Pclass","Sex","Age"]
test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())
test_data["Sex"] = test_data["Sex"].apply(lambda sex:1 if sex=="male" else 0)
passengerID = test_data["PassengerId"].values
test_data = test_data[features].values


#Regression Model
model = LogisticRegression()
model.fit(train_data,survived)
predicted = model.predict(test_data)
output = pd.DataFrame(columns=['PassengerId','Survived'])
output['PassengerId'] = passengerID
output['Survived'] = predicted.astype(int)
output.to_csv('logisticRegressionSubmit.csv', index=False)

