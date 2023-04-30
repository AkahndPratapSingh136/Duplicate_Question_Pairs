#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[34]:


df=pd.read_csv('train.csv')


# In[35]:


df.shape


# In[36]:


df.head()


# In[39]:


new_df=df.sample(30000)


# In[44]:


# Count of null values
new_df.isnull().sum()


# In[45]:


# Count of empty rows
new_df.duplicated().sum()


# In[46]:


ques_df=new_df[['question1','question2']]
ques_df.head()


# In[47]:


# Merge Texts
from sklearn.feature_extraction.text import CountVectorizer
questions=list(ques_df['question1'])+list(ques_df['question2'])
cv=CountVectorizer(max_features=3000) #Most used 3000 words feature
q1_arr,q2_arr=np.vsplit(cv.fit_transform(questions).toarray(),2) #We get BOW for every questions


# In[48]:


# Creating a new dataframe temp_df consisting 
temp_df1=pd.DataFrame(q1_arr, index=ques_df.index)
temp_df2=pd.DataFrame(q2_arr, index=ques_df.index)
temp_df=pd.concat([temp_df1,temp_df2],axis=1)
temp_df.shape


# In[50]:


temp_df


# In[51]:


temp_df['is_duplicate']=new_df['is_duplicate']


# In[55]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(temp_df.iloc[:,0:-1].values,temp_df.iloc[:,-1].values,test_size=0.2,random_state=1)


# In[61]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
accuracy_score(y_test,y_pred)*100


# In[65]:


pip install xgboost


# In[66]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:




