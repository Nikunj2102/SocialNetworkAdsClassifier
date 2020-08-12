#!/usr/bin/env python
# coding: utf-8

# IMPORT STATEMENTS

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report


# IMPORT DATASET

# In[2]:


datasets = pd.read_csv('Social_Network_Ads.csv')
datasets.describe()


# In[3]:


X_ref = datasets.iloc[:,[2,3]].head()


# In[4]:


Y_ref = datasets.iloc[:,4].head(10)


# In[5]:


X = datasets.iloc[:, [2,3]].values
Y = datasets.iloc[:, 4].values


# In[6]:


#Splitting the dataset into testing and training datasets
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y, test_size = 0.25, random_state = 0)


# Feature scaling

# In[7]:


sc_X = StandardScaler()


# In[8]:


X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)


# FITTING THE CLASSIFIER INTO THE TRAINING SET

# In[9]:


classifier = RandomForestClassifier(n_estimators = 200, criterion = 'gini', random_state = 0)
classifier.fit(X_Train,Y_Train)


# ---------------------------------------

# PREDICTING THE TEST SET RESULTS

# In[10]:


Y_Pred = classifier.predict(X_Test)


# ----

# MAKING CONFUSION MATRIX

# In[11]:


cm = confusion_matrix(Y_Test, Y_Pred)
print(cm)


# In[12]:


X_Set, Y_Set = X_Test, Y_Test


# In[13]:


X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))


plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'yellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[14]:


print(classification_report(Y_Test,Y_Pred))

