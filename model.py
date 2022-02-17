from load_csv import load_data_csv
from get_basic_info import check_basic_information,check_null_value,check_shape, check_top_column, to_save_data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Loading the training dataset
train_df = load_data_csv('G:\\Kaggle_compitation\\Logistic Regression\\Dataset\\train.csv')
print(f"The training data is :- \n {train_df.head()}")

# Loading the testing dataset

test_df = load_data_csv('G:\\Kaggle_compitation\\Logistic Regression\\Dataset\\test.csv')
print(f"The testing data is :- \n {test_df.head()}")

# shape of the training and testing data
print(f"The shape of the training data is {check_shape(train_df)}")
print(f"The shape of the testing data is {check_shape(test_df)}")

# getting the basic information about the data and its type
print(f"The information in the training data is {check_basic_information(train_df)}")
print(f"The information in the testing data is {check_basic_information(test_df)}")

# Getting the missing value in the dataset
print(f"The missing value in the training data is {check_null_value(train_df)}")
print(f"The missing value in the testing data is {check_null_value(test_df)}")

# Getting the percentage of missing value in Age column.

print('Percent of missing "Age" records is %.2f%%' %((train_df['Age'].isnull().sum()/train_df.shape[0])*100))

'''ax = train_df['Age'].hist(bins = 5,density = True, stacked = True, color = 'teal', alpha = 0.6)
train_df['Age'].plot(kind = 'density', color = 'teal')
ax.set(xlabel= 'Age')
plt.xlim(-10,85)
plt.show()
'''
# mean age
print('The mean of "Age" is %.2f' %(train_df["Age"].mean(skipna=True)))
# median age
print('The median of "Age" is %.2f' %(train_df["Age"].median(skipna=True)))


# Getting missing value in Cabin columns
print('Percent of missing "Cabin" records is %.2f%%' %((train_df['Cabin'].isnull().sum()/train_df.shape[0])*100))
# From the output we can see that this column have 77.10% missing value so we need to ignore it.

# Getting missing value in Embarked columns
print('Percent of missing "Cabin" records is %.2f%%' %((train_df['Embarked'].isnull().sum()/train_df.shape[0])*100))
#Percent of missing "Embarked" records is 0.22% to replace this we can use mode.
print('Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):')
print(train_df['Embarked'].value_counts())
'''sns.countplot(x='Embarked', data=train_df, palette='Set2')
plt.show()
'''
print('The most common boarding port of embarkation is %s.' %train_df['Embarked'].value_counts().idxmax())
#By far the most passengers boarded in Southhampton, so we'll impute those 2 NaN's w/ "S".

# Replacing all the missing value
train_data = train_df.copy()
train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)
# Checking the missing value
print(f"The missing value in the training data is {check_null_value(train_data)}")

print(f"The top column in the training data is {check_top_column(train_data)}")
# print(f"The top column in the testing data is {check_top_column(test_df)}")

'''plt.figure(figsize=(15,8))
ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_df["Age"].plot(kind='density', color='teal')
ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)
train_data["Age"].plot(kind='density', color='orange')
ax.legend(['Raw Age', 'Adjusted Age'])
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()'''

## Create categorical variable for traveling alone
train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)


#create categorical variables and drop some variables
training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
print(f"the top column is the data is {check_top_column(final_train)}")

  

# Now we can apply same thing to test data.
print(f"The missing value in the testing data is {check_null_value(test_df)}")

test_data = test_df.copy()
test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)

test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
print(f"The top column in the data is {check_top_column(final_test)}")
print(f"The missing value in the testing data is {check_null_value(final_test)}")

# EDA
'''plt.figure(figsize=(15,8))
ax = sns.kdeplot(final_train["Age"][final_train.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(final_train["Age"][final_train.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()

plt.figure(figsize=(20,8))
avg_survival_byage = final_train[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()
g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")
plt.show()
'''
# Replacing the minor child and major child
final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)

final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)
'''
#Exploration of Fare
plt.figure(figsize=(15,8))
ax = sns.kdeplot(final_train["Fare"][final_train.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(final_train["Fare"][final_train.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
ax.set(xlabel='Fare')
plt.xlim(-20,200)
plt.show()

#Exploration of Passenger Class
sns.barplot('Pclass', 'Survived', data=train_df, color="darkturquoise")
plt.show()

#Exploration of Embarked Port
sns.barplot('Embarked', 'Survived', data=train_df, color="teal")
plt.show()

#Exploration of Traveling Alone vs. With Family
sns.barplot('TravelAlone', 'Survived', data=final_train, color="mediumturquoise")
plt.show()

# Exploration of Gender Variable
sns.barplot('Sex', 'Survived', data=train_df, color="aquamarine")
plt.show()'''

final_train.to_csv('final_train.csv')

final_test.to_csv('final_test.csv')
