#!/usr/bin/env python
# coding: utf-8

# # Random Forest Model Results using Python Script

# In[8]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import pickle;
import os;

#df=pd.read_csv("./Data/master_data.csv")
#print(df.head())


# ## Select Model. 
# * Select 1 for 30 Days model
# * Select 2 for 30 Days model
# * Select 3 for 30 Days model

# In[9]:


model=int(input("Select 1 for 30 Days model, 2 for 60 Days model and 3 for 90 Days Model"))
dataPath="C:/Users/Nageswararao/Downloads/Python NoteBooks First-Batch/Python NoteBooks First-Batch/Data"


# ## Read appropriate Model data

# In[10]:


if model == 1:
   df=pd.read_csv(dataPath+"/model30data.csv")
   df1=df.iloc[:,1:]
elif model == 2:
   df=pd.read_csv(dataPath+"/model60data.csv")
   df1=df.iloc[:,1:]
else:
   df=pd.read_csv(dataPath+"/model90data.csv")
   df1=df.iloc[:,1:]


# In[11]:


#print(df1.shape)
df2=df1.iloc[:,:133]
df2=df2.dropna()
#print(df2.tail())
if model == 1:
   X= df2.drop('numInpatientInLast30',axis=1)
   y=df2['numInpatientInLast30']
elif model == 2:
   X= df2.drop('numInpatientInLast60',axis=1)
   y=df2['numInpatientInLast60']
else:
   X= df2.drop('numInpatientInLast90',axis=1)
   y=df2['numInpatientInLast90']


# In[15]:


rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
rf.fit(X, y)
# print (X.head())
from sklearn.metrics import accuracy_score
predicted = rf.predict(X)
accuracy = accuracy_score(y, predicted)
print("Predicted ?? ", rf.predict(pd.DataFrame([[2,1,9,2,5,1,1,2,9,3,9,9,9,9,9,9,9,9,9,9,9,9,9,1,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,4,1,3,1,1,1,0,1,1,1,1,1,1]])));


# ## OOB Score and Mean Accuracy Score

# In[9]:


print('Out-of-bag score estimate:', rf.oob_score_)
print('Mean accuracy score:', accuracy)


# ##  Data Accuracy

# In[6]:


pred=rf.predict(X)
accuracy=accuracy_score(y,pred)
print("Model Accuracy :",accuracy)
print("\nConfusion Matrix of the data\n")
print(metrics.confusion_matrix(y,pred))


file=open(dataPath+'/model.pkl','wb')
print (os.path.realpath(dataPath+'/model.pkl'))
pickle.dump(rf, file)
file.close()
print("PKL is dumped");


# In[ ]:




