#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


raw_mail_data=pd.read_csv(r'C:\Users\HP\Downloads\mail_data.csv')


# In[4]:


print(raw_mail_data)


# In[5]:


#replacing null values with null strings
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[6]:


mail_data.head()


# In[7]:


mail_data.shape


# In[8]:


#Label Encoding(Spam as 0, Ham as 1)
mail_data.loc[mail_data['Category']=='spam','Category',]=0
mail_data.loc[mail_data['Category']=='ham','Category',]=1


# In[9]:


#Separating data as texts and label
X=mail_data['Message']
Y=mail_data['Category']


# In[10]:


print(X)
print(Y)


# In[12]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)


# In[13]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[28]:


feature_extraction=TfidfVectorizer(min_df=1,stop_words='english')
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)


# In[29]:


Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')


# In[30]:


print(X_train)


# In[31]:


print(X_train_features)


# In[32]:


#Training model on Logistic Regression
model=LogisticRegression()
model.fit(X_train_features,Y_train)


# In[33]:


prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)
print('Accuracy:',accuracy_on_training_data)


# In[34]:


prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
print('Accuracy:',accuracy_on_test_data)


# In[35]:


#Building a predictive system
input_mail=["WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]
input_data_features=feature_extraction.transform(input_mail)


# In[36]:


#Making prediction
prediction=model.predict(input_data_features)
print(prediction)
if(prediction[0]==1):
    print('Ham mail')
else:
    print('Spam mail')


# In[ ]:




