#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np # linear algebra
import pandas as pd # data processing
import re
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


from sklearn.model_selection import train_test_split
df = pd.read_csv("data.csv")
df.head()

X = df.iloc[:,1]
y = df.iloc[:,2:]


# In[3]:


comments = list(X)
comments[:4]


# In[4]:


comments = [re.sub(r'(\')', "", w.lower()) for w in comments]
comments[:4]


# In[5]:


comments = [re.sub('[^A-Za-z ]+', ' ', w) for w in comments]
comments[:4]


# In[6]:


comments = [re.sub( '\s+', ' ', w ).strip() for w in comments]
comments[:4]


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(comments, y, test_size=0.2, random_state=123)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

print(y.apply(lambda x:  x.value_counts()/x.value_counts().sum()))
print(y_train.apply(lambda x:  x.value_counts()/x.value_counts().sum()))
print(y_test.apply(lambda x:  x.value_counts()/x.value_counts().sum()))
print(y_val.apply(lambda x:  x.value_counts()/x.value_counts().sum()))


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect = CountVectorizer(decode_error='ignore',stop_words='english')
train_dtm=vect.fit_transform(X_train)
train_dtm
#vect2 = TfidfVectorizer(decode_error='ignore',stop_words='english')
#train_emails_tfid=vect2.fit_transform(comments)


# In[17]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=0.01)
nb.fit(train_dtm, y_train.severe_toxic)


# In[18]:


y_pred_nb = nb.predict(train_dtm)
from sklearn import metrics
nbAcc=metrics.accuracy_score(y_train.severe_toxic, y_pred_nb)
nbAcc


# In[19]:


test_dtm=vect.transform(X_test)
y_pred_nb_test = nb.predict(test_dtm)


# In[20]:


nbAcc_test = metrics.accuracy_score(y_test.severe_toxic, y_pred_nb_test)
nbAcc_test


# In[29]:


print(metrics.classification_report(y_test.severe_toxic, y_pred_nb_test))


# In[30]:


nb = MultinomialNB(alpha=1.0, fit_prior=False)
nb.fit(train_dtm, y_train.severe_toxic)


# In[31]:


y_pred_nb = nb.predict(train_dtm)
from sklearn import metrics
nbAcc1=metrics.accuracy_score(y_train.severe_toxic, y_pred_nb)
nbAcc1


# In[36]:


test_dtm=vect.transform(X_test)
y_pred_nb_test1 = nb.predict(test_dtm)


# In[37]:


nbAcc_test12 = metrics.accuracy_score(y_test.severe_toxic, y_pred_nb_test1)
nbAcc_test12

