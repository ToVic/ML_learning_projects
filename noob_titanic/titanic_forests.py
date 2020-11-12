import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
raw = pd.read_csv('train.csv')

#print(raw.columns)

#['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

#lets find relevant data

data = raw[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]
data['Sex'] = data['Sex'].map({'male':0,'female':1})
data = data.dropna().reset_index()
data['Embarked'] = data['Embarked'].map({'C':1,'Q':2,'S':3})
labels = data['Survived']
dataa = data.drop('Survived', axis = 1)

train_data, test_data, train_labels, test_labels = train_test_split(dataa, labels, random_state = 42, test_size = 0.15)
forest = RandomForestClassifier(random_state=42)
forest.fit(train_data, train_labels)

print(forest.score(test_data, test_labels))

letscheck = pd.read_csv('test.csv')
prediction = letscheck[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
prediction['Sex'] = prediction['Sex'].map({'male':0,'female':1})
prediction['Embarked'] = prediction['Embarked'].map({'C':1,'Q':2,'S':3})
prediction = prediction.fillna(0).reset_index()
result = forest.predict(prediction)


results = pd.DataFrame()
results['PassengerId'] = letscheck['PassengerId']
results['Survived'] = pd.Series(result)
results.to_csv('submission.csv')
