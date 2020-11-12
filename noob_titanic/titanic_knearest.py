import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

raw = pd.read_csv('train.csv')
raw['Age'].fillna(raw['Age'].mean(), inplace = True)
raw['Embarked'].fillna('X', inplace = True)

data = raw[['Pclass', 'Sex', 'Age']]
labels = raw['Survived']
data['Sex'] = data['Sex'].map({'male':0,'female':1})
#data['Embarked'] = data['Embarked'].map({'X' : 0, 'C': 1, 'Q' : 2, 'S' : 3})

train_data, val_data, train_labels, val_labels =\
            train_test_split(data, labels, test_size = 0.2, random_state = 100)

classifier = KNeighborsClassifier()
classifier.fit(train_data, train_labels)
print(classifier.score(val_data, val_labels))
