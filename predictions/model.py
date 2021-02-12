# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import 
import pandas as pd
import numpy as np


# %%
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# %%
def prepare_data(X):
    X['is_adult'] = X['Age']
    for i,val in enumerate(X['Sex']):
        if X['Sex'][i] == 'male':
            X['Sex'][i] = 1
        else:
            X['Sex'][i] = 0
    for i,val in enumerate(X['Age']):
        if X['Age'][i] < 18:
            X['is_adult'][i] = 1
        else:
            X['is_adult'][i] = 0
    median_age = X['Age'].median()
    X['Age'] = X['Age'].fillna(median_age)
    median_fare = X['Fare'].median()
    X['Fare'] = X['Fare'].fillna(median_age)
    return X


# %%
y = train['Survived']
X = train.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'], 1)
X = prepare_data(X)
# lr = LinearRegression()
model = KNeighborsClassifier(5, n_jobs=12)


# %%
X


# %%
test = test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], 1)
test = prepare_data(test)
test


# %%
model.fit(X,y)
prediction = pd.DataFrame(abs(model.predict(test).round()))
prediction['PassengerId'] = test['PassengerId']


# %%
import datetime
datetime.datetime.now()
timestamp = f'predictions/prediction-{str(datetime.datetime.now())}.csv'
prediction.to_csv(timestamp, encoding='utf-8', index=False)


