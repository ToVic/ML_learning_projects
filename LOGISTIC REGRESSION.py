import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sns.set()

passengers = pd.read_csv('train.csv')


passengers['Sex'] = passengers['Sex'].map({'male' : 0, 'female' : 1})


passengers['Age'].fillna(passengers['Age'].mean(), inplace=True)


passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)


passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)


features = passengers[['Sex','Age','FirstClass','SecondClass']]
survival = passengers['Survived']

train_feat, test_feat, train_labels, test_labels = train_test_split(features, survival, )


scaler = StandardScaler()
train_feat = scaler.fit_transform(train_feat)
test_feat = scaler.transform(test_feat)



model = LogisticRegression()
model.fit(train_feat, train_labels)

# Score the model on the train data
#print(model.score(train_feat,train_labels))

# Score the model on the test data
#print(model.score(test_feat, test_labels))

# Analyze the coefficients
#print(model.coef_)

stest = pd.read_csv('test.csv')
tester = stest.copy()
tester['Sex'] = tester['Sex'].map({'male' : 0, 'female' : 1})
tester['Age'].fillna((tester['Age'].mean()+passengers['Age'].mean())/2, inplace=True)
tester['FirstClass'] = tester['Pclass'].apply(lambda x: 1 if x == 1 else 0)
tester['SecondClass'] = tester['Pclass'].apply(lambda x: 1 if x == 2 else 0)
featuress = tester[['Sex','Age','FirstClass','SecondClass']]
featuress = scaler.transform(featuress)
resulto = model.predict(featuress)


results = pd.DataFrame()
results['PassengerId'] = stest['PassengerId']
results['Survived'] = pd.Series(resulto)
results.to_csv('submission3.csv')
