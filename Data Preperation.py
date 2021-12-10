#!/usr/bin/env python
# coding: utf-8

# # Data Pre-processing for Multi-Label text classification

# In[4]:


# Importing required libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# In[5]:


# Load data
df = pd.read_csv("data/data.csv")
df.head()


# In[6]:


# separate explanatory and dependent variables
X = df.iloc[:,1]
y = df.iloc[:,2:]


# In[7]:


# split for cross-validation (train-60%, validation 20% and test 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=123)


# In[8]:


# Check if the class proportions are maintained in the splits
print(y.apply(lambda x:  x.value_counts()/x.value_counts().sum()))
print(y_train.apply(lambda x:  x.value_counts()/x.value_counts().sum()))
print(y_test.apply(lambda x:  x.value_counts()/x.value_counts().sum()))
print(y_val.apply(lambda x:  x.value_counts()/x.value_counts().sum()))

