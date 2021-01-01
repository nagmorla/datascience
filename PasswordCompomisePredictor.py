#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle;
import os;
import numpy as np;
import pandas as pd;
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

print ("hello");


# In[13]:


dataPath="C:/Users/Nageswararao/Downloads/Python NoteBooks First-Batch/Python NoteBooks First-Batch/Data"
df=pd.read_csv(dataPath+"/pwds.csv")
X= df.drop('COMPROMISED',axis=1)
y=df['COMPROMISED']
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)


# In[14]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

rf.fit(X_train, y_train)


# In[26]:



predicted = rf.predict(X)
print ("Pridicted on X")
print (predicted)

y_pred=rf.predict(X_test)
print ("Pridected on X Test Data")
print (y_pred)

accuracy = accuracy_score(y, predicted)
print ("Accuracy", accuracy)

print("Confusion Matix",metrics.confusion_matrix(y,predicted))

feature_imp = pd.Series(rf.feature_importances_,index=["LENGTH","UPPER","LOWER","NUMBERS","SPECIAL","AGE"]).sort_values(ascending=False)
print (feature_imp)

print ("My Password can be compromised?",rf.predict([[13,3,8,1,1,10]]))

