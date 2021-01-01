#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
from flask import Flask, request, jsonify

import pandas as pd
import pickle


# In[15]:


app = Flask(__name__)
dataPath="C:/Users/Nageswararao/Downloads/Python NoteBooks First-Batch/Python NoteBooks First-Batch/Data"
model = pickle.load(open(dataPath+'/model.pkl','rb'))


# In[24]:


@app.route('/api',methods=['POST'])
def predictfun():
    print ("serving request")
    data = request.get_json(force=True)
    print ("Exp Data::",data['exp'])
    arr=np.array(data['exp']);
    print (arr)
    arr1=pd.DataFrame([[arr]]);
    print ("DataFrame ** ", arr1)
    try:
        print ("---------------------------")
        print (model.predict([[2,1,9,2,5,1,1,2,9,3,9,9,9,9,9,9,9,9,9,9,9,9,9,1,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,4,1,3,1,1,1,0,1,1,1,1,1,1]]))
        print ("---------------------------")
    except Exception as e:
        print ("Error while predicting static data.", e);
    prediction = model.predict(arr1);
    print (prediction);
    print ("--------------------------")
    output = prediction[0]
    return jsonify(output)


# In[12]:


if __name__ == '__main__':
    app.run(port=8080)

