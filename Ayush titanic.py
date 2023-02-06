#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[10]:


X=[4,5,10,4,10,8,3,2,7,3]
Y=[23,27,22,30,21,23,27,29,30,25]
data=list(zip(X,Y))
data


# In[11]:


model=KMeans(n_clusters=2)
model.fit(data)


# In[12]:


plt.scatter(X,Y,c=model.labels_)
plt.show()


# In[ ]:





# In[41]:


data = pd.read_csv("E:/titanic_dataset.csv")
data=data.dropna()
X=data["Age"]
Y=data["Fare"]


# In[44]:


data=list(zip(X.values,Y.values))
model=KMeans(n_clusters=4)
model.fit(data)


# In[45]:


plt.scatter(X,Y, c=model.labels_)
plt.show()

