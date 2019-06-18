import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

loans = pd.read_csv('loan_data.csv')

# Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome
plt.figure(figsize = (11,7))
loans[loans['credit.policy'] == 1]['fico'].hist(alpha = 0.5, bins = 30,color='blue',label='Credit Policy = 1')
loans[loans['credit.policy'] == 0]['fico'].hist(alpha = 0.5, bins = 30,color='red',label='Credit Policy = 0')
plt.legend()
plt.xlabel('FICO')

# Create a similar figure, except this time select by the not.fully.paid column
plt.figure(figsize = (11,7))
loans[loans['not.fully.paid'] == 1]['fico'].hist(alpha = 0.5, bins = 30, color = 'blue', label = 'not.fully.paid = 1')
loans[loans['not.fully.paid'] == 0]['fico'].hist(alpha = 0.5, bins = 30, color = 'red', label = 'not.fully.paid = 0')
plt.legend()
plt.xlabel('FICO')

# Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid
plt.figure(figsize = (11,7))
sns.countplot(x = 'purpose',data = loans, hue='not.fully.paid')

# Observe the trend between FICO score and interest rate in a jointplot
sns.jointplot(x='fico',y='int.rate',data=loans)

# Create lmplots to see if the trend differed between not.fully.paid and credit.policy
plt.figure(figsize=(11,7))
sns.lmplot(x='fico',y='int.rate',data=loans,hue='credit.policy', col='not.fully.paid')

# Transform categorical data (purpose column) using dummy variables so sklearn will be able to understand them
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)
Y = final_data['not.fully.paid']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=101)

# Train model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)

# Predictions and evaluation of model
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))

# Trying a random forest model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 600)
rfc.fit(X_train,Y_train)
predict2 = rfc.predict(X_test)
print(classification_report(Y_test,predict2))
print(confusion_matrix(Y_test,predict2))
