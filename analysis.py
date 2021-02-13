# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# %%
df = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
test['Fare'] = test['Fare'].fillna(test['Fare'].median())


# %%
df.shape


# %%
df.head()


# %%
df.describe()


# %%
df.isna().sum()


# %%
print(df['Survived'].value_counts())
sns.countplot(x=df['Survived'])


# %%
df['Sex'] = df['Sex'].astype('category')
print(df['Sex'])


# %%
def group_age_by_ten(row):
    if np.isnan(row.Age):
        return 'undefined'
    return int(row.Age/10)

df['AgeGroupsTen'] = df.apply(group_age_by_ten, axis=1)
test['AgeGroupsTen'] = test.apply(group_age_by_ten, axis=1)
df['AgeGroupsTen'] = df['AgeGroupsTen'].astype('category')
test['AgeGroupsTen'] = test['AgeGroupsTen'].astype('category')

def group_age_by_title (row):
    if row.Age < 16:
        return 1
    if 16 <= row.Age <= 55:
        return 2
    if row.Age > 55:
        return 3
    return 0

df['AgeGroupsTitle'] = df.apply(group_age_by_title, axis=1)
test['AgeGroupsTitle'] = test.apply(group_age_by_title, axis=1)
df['AgeGroupsTitle'] = df['AgeGroupsTitle'].astype('category').cat.rename_categories(['undefined', 'kid', 'adult', 'elder'])
test['AgeGroupsTitle'] = test['AgeGroupsTitle'].astype('category').cat.rename_categories(['undefined', 'kid', 'adult', 'elder'])

fig, axes = plt.subplots(1,2, figsize=(15,5))
sns.countplot(x='AgeGroupsTen',  hue='Survived',data=df, ax = axes[0])
sns.countplot(x='AgeGroupsTitle',hue='Survived', data=df, ax = axes[1])


# %%
#dropping ticket, no valuable data found on it
df = df.drop('Ticket', axis=1)
test = test.drop('Ticket', axis=1)


# %%
df['Fare'] = (df['Fare']/10).astype(int)
test['Fare'] = (test['Fare']/10).astype(int)


# %%
plt.figure(figsize=(30,6))
sns.countplot(x='Fare', hue='Survived', data=df)
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(30,6))
df.groupby('Fare').mean()['Survived'].plot()
plt.show()


# %%
plt.figure(figsize=(8,30))
plt.title("survival rate for each Cabin")
sns.countplot(y='Cabin', hue='Survived', data=df.sort_values('Cabin'), orient='h')


# %%



# %%
def group_cabin(row):
    if pd.isna(row.Cabin):
        return 0
    if df['Cabin'].value_counts()[row.Cabin] > 1:
        return 1
    return 2
def group_cabin_test(row):
    if pd.isna(row.Cabin):
        return 0
    if test['Cabin'].value_counts()[row.Cabin] > 1:
        return 1
    return 2

df['Cabin'] = df.apply(group_cabin, axis=1)
test['Cabin'] = test.apply(group_cabin_test, axis=1)
df['Cabin'] = df['Cabin'].astype('category').cat.rename_categories(['unidentified', 'group', 'solo'])
test['Cabin'] = test['Cabin'].astype('category').cat.rename_categories(['unidentified', 'group', 'solo'])

g = df.groupby('Cabin').mean()['Survived'].plot.bar()
g.set_xticklabels(g.get_xticklabels(), rotation=0)
plt.ylim(top=1)
plt.show()


# %%
print(df['Embarked'].value_counts())
print('Empties before filling: ', df['Embarked'].isna().sum())
df['Embarked'].fillna('S', inplace=True)
test['Embarked'].fillna('S', inplace=True)
print('Empities after filling: ', df['Embarked'].isna().sum())
df['Embarked'] = df['Embarked'].astype('category')
test['Embarked'] = test['Embarked'].astype('category')


# %%
columns = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Cabin', 'Embarked', 'Fare', 'AgeGroupsTitle']

fig, axes = plt.subplots(3,3,constrained_layout=True, figsize=(20,10))

for i, col in enumerate(columns):
    if i == 8:
        break
    sns.countplot(x = df[col], hue=df['Survived'], ax = axes[int(i/3), int(i%3)])
axes[int(i/3), int(i%3)].legend(loc='upper right')


# %%
df.groupby(['AgeGroupsTitle', 'Sex', 'Survived'])['PassengerId'].count().unstack('Survived').plot(kind='bar', rot=0, figsize=(20,4))


# %%
df.groupby(['Fare', 'Pclass'])['Fare'].count().unstack('Pclass').plot(kind='bar', rot=0, stacked=True, figsize=(20,4)).set_xlabel('Fare (10$)')


# %%
plt.figure(figsize=(20,5))
plt.title('Embarked port for each class')
sns.countplot(x='Pclass', hue='Embarked', data=df)
plt.figure(figsize=(20,5))
plt.title('Embarked port for each fare')
sns.countplot(x='Fare', hue='Embarked', data=df)


# %%
fig, axes = plt.subplots(1,4, figsize=(30,5))
big_ax = fig.add_subplot(1,2,1)
fig.delaxes(axes[0])
fig.delaxes(axes[1])
big_ax.set_title("The First class's survival rate for each fare")
sns.countplot(x = 'Fare', hue='Survived', data=df[df['Pclass'] == 1], ax=big_ax)
axes[2].set_title("The second class's survival rate for each fare")
sns.countplot(x = 'Fare', hue='Survived', data=df[df['Pclass'] == 2], ax=axes[2])
axes[3].set_title("The Third class's survival rate for each fare")
sns.countplot(x = 'Fare', hue='Survived', data=df[df['Pclass'] == 3], ax=axes[3])


# %%
df2 = df[(df['SibSp'] > 0) | (df['Parch'] > 0)]
sns.heatmap(df2[['SibSp', 'Parch']].corr(), annot=True, square=True) 


# %%
df2[(df['SibSp'] > 0) & (df['Parch'] > 0)]
df2['family'] = df2['SibSp'] + df2['Parch']
sns.countplot(x='family', hue='Survived', data=df2)


# %%
df.head()


# %%
def prepare_data(data):
    
    data['sex'] = data['PassengerId']
    data['cabin'] = data['PassengerId']
    data['embarked'] = data['PassengerId']
    data['ageGroupsTitle'] = data['PassengerId']

    for i,val in enumerate(data['Sex']):
        if data['Sex'][i] == 'male':
            data['sex'][i] = 1
        else:
            data['sex'][i] = 0
    # print(data['Age'].median())
    data['Age'] = data['Age'].fillna(data['Age'].median())
    
    for i, value in enumerate(data['Cabin']):
        if value == 'unidentified':
            data['cabin'][i] = 0
        if value == 'solo':
            data['cabin'][i] = 1
        if value == 'group':
            data['cabin'][i] = 2
    
    for i, value in enumerate(data['Embarked']):
        if value == 'S':
            data['embarked'][i] = 0
        if value == 'C':
            data['embarked'][i] = 1
        if value == 'Q':
            data['embarked'][i] = 2

    for i, value in enumerate(data['AgeGroupsTen']):
        if value == 'undefined':
            data['AgeGroupsTen'][i] = 0

    for i, value in enumerate(data['AgeGroupsTitle']):
        if value == 'undefined':
            data['ageGroupsTitle'][i] = 0
        if value == 'kid':
            data['ageGroupsTitle'][i] = 1
        if value == 'adult':
            data['ageGroupsTitle'][i] = 2
        if value == 'elder':
            data['ageGroupsTitle'][i] = 3
     
    data['Fare'].fillna(data['Fare'].median()/10)

    data = data.drop(['Sex', 'Cabin', 'Embarked', 'AgeGroupsTitle'], axis=1)

    return data


# %%
df_train_X = df
df_train_y = df['Survived']
df_train_X = df_train_X.drop(['Survived', 'Name'], axis=1)

# df_train_X
df_train_x = prepare_data(df_train_X)


# %%
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

clf_splitted = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))


# %%
x_train,x_test,y_train, y_test = train_test_split(df_train_x,df_train_y, test_size=0.25)


# %%
clf_splitted.fit(x_train, y_train)
prediction = clf_splitted.predict(x_test)
(prediction == y_test).sum()/len(prediction)


# %%



# %%
clf.fit(df_train_x, df_train_y)


# %%



# %%
df_test_x = prepare_data(test)
df_test_x = df_test_x.drop('Name', axis=1)
df_test_x


# %%
prediction_submit = clf.predict(df_test_x)
prediction_submit


# %%
prediction = pd.DataFrame()
prediction['PassengerId'] = df_test_x['PassengerId']
prediction['Survived'] = prediction_submit


# %%
import datetime
datetime.datetime.now()
timestamp = f'predictions/prediction-{str(datetime.datetime.now())}.csv'
prediction.to_csv(timestamp, encoding='utf-8', index=False)


