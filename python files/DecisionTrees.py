#!/usr/bin/env python
# coding: utf-8

# IMPORT STATEMENTS

# In[143]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix

from matplotlib.colors import ListedColormap

from sklearn.metrics import classification_report


# IMPORT DATASET

# In[144]:


datasets = pd.read_csv('Social_Network_Ads.csv')
datasets.describe()


# In[145]:


X_ref = datasets.iloc[:,[2,3]].head()


# In[146]:


Y_ref = datasets.iloc[:,4].head(10)


# In[147]:


X = datasets.iloc[:, [2,3]].values
Y = datasets.iloc[:, 4].values


# In[148]:


#Splitting the dataset into testing and training datasets
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y, test_size = 0.25, random_state = 0)


# Feature scaling

# In[149]:


sc_X = StandardScaler()


# In[150]:


X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)


# FITTING THE CLASSIFIER INTO THE TRAINING SET

# In[151]:


classifier = tree.DecisionTreeClassifier()
classifierBagging = BaggingClassifier(n_estimators = 125 , random_state = 0)
classifier.fit(X_Train,Y_Train)
classifierBagging.fit(X_Train,Y_Train)


# ---------------------------------------

# PREDICTING THE TEST SET RESULTS

# In[152]:


Y_Pred = classifier.predict(X_Test)
Y_Pred_Bagging = classifierBagging.predict(X_Test)


# ----

# MAKING CONFUSION MATRIX

# In[153]:


cm = confusion_matrix(Y_Test, Y_Pred)
cm_bagging = confusion_matrix(Y_Test,Y_Pred_Bagging)
print("W/O", cm,"\n")
print("W:", cm_bagging)


# In[154]:


X_Set, Y_Set = X_Test, Y_Test


# In[155]:


X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))


plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'yellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[156]:


plt.contourf(X1, X2, classifierBagging.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'yellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Bagged Decision Tree (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[157]:


print("With Bagging: ",classification_report(Y_Test,Y_Pred_Bagging),"\n")
print("Without Bagging: ",classification_report(Y_Test,Y_Pred))


# In[ ]:




