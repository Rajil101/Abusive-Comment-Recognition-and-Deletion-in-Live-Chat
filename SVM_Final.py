#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing required libraries
import numpy as np # linear algebra
import pandas as pd # data processing 
import re
from sklearn.feature_extraction.text import CountVectorizer


# In[4]:


from sklearn.model_selection import train_test_split
df = pd.read_csv("data.csv")
df.head()

X = df.iloc[:,1]
y = df.iloc[:,2:]


# In[5]:


comments = list(X)
comments[:4]


# In[6]:


comments = [re.sub(r'(\')', "", w.lower()) for w in comments]
comments[:4]


# In[7]:


comments = [re.sub('[^A-Za-z ]+', ' ', w) for w in comments]
comments[:4]


# In[8]:


comments = [re.sub( '\s+', ' ', w ).strip() for w in comments]
comments[:4]


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(comments, y, test_size=0.2, random_state=123)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

print(y.apply(lambda x:  x.value_counts()/x.value_counts().sum()))
print(y_train.apply(lambda x:  x.value_counts()/x.value_counts().sum()))
print(y_test.apply(lambda x:  x.value_counts()/x.value_counts().sum()))
print(y_val.apply(lambda x:  x.value_counts()/x.value_counts().sum()))


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect = CountVectorizer(decode_error='ignore',stop_words='english')
train_dtm=vect.fit_transform(X_train)
train_dtm
#vect2 = TfidfVectorizer(decode_error='ignore',stop_words='english')
#train_emails_tfid=vect2.fit_transform(comments)


# In[11]:


from sklearn import svm 
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(train_dtm, y_train.severe_toxic)


# In[12]:


y_pred= clf.predict(train_dtm)
y_pred


# In[13]:


from sklearn import metrics
nbAcc1=metrics.accuracy_score(y_train.severe_toxic, y_pred)
nbAcc1


# In[ ]:


test = np.array([y_train.severe_toxic])
y__test = test.reshape(1,-1)
pred_test = clf.predict(y__test)
nbAcc_test1 = metrics.accuracy_score(y_test.severe_toxic, pred_test)
nbAcc_test1


# In[ ]:


from sklearn import svm 
clf = svm.SVC(C=5.0,  cache_size=100,decision_function_shape='ovo')
clf.fit(train_dtm, y_train.severe_toxic)


# In[ ]:


y_pred= clf.predict(train_dtm)
y_pred


# In[ ]:


from sklearn import metrics
nbAcc12=metrics.accuracy_score(y_train.severe_toxic, y_pred)
nbAcc12


# In[ ]:


test = np.array([y_train.severe_toxic])
y__test = test.reshape(1, -1)
pred_test = clf.predict(y__test)
nbAcc_test2 = metrics.accuracy_score(y_test.severe_toxic, pred_test)
nbAcc_test2

