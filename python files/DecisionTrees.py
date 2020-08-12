#!/usr/bin/env python
# coding: utf-8

# IMPORT STATEMENTS

# In[20]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statistics import mean
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix

from matplotlib.colors import ListedColormap

from sklearn.metrics import classification_report


# IMPORT DATASET

# In[21]:


datasets = pd.read_csv('Social_Network_Ads.csv')
datasets.describe()


# In[22]:


X = datasets.iloc[:, [2,3]].values
Y = datasets.iloc[:, 4].values

kf = KFold(n_splits = 5)
f1score = []
f1scoreBagging = []
for train_index , test_index in kf.split(X):
    X_Train,X_Test = X[train_index],X[test_index]
    Y_Train,Y_Test = Y[train_index],Y[test_index]
    
    #Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform(X_Test)
    
    #FITTING THE CLASSIFIER INTO THE TRAINING SET
    
    classifier = tree.DecisionTreeClassifier()
    classifierBagging = BaggingClassifier(n_estimators = 125 , random_state = 0)
    classifier.fit(X_Train,Y_Train)
    classifierBagging.fit(X_Train,Y_Train)
    
    #PREDICTING THE TEST SET RESULTS
    Y_Pred = classifier.predict(X_Test)
    Y_Pred_Bagging = classifierBagging.predict(X_Test)
    
    #Making confusion matrix
    cm = confusion_matrix(Y_Test, Y_Pred)
    cm_bagging = confusion_matrix(Y_Test,Y_Pred_Bagging)
    print("W/O", cm,"\n")
    print("W:", cm_bagging)
    
    X_Set, Y_Set = X_Test, Y_Test
    
    #Plotting
    X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))

    #Without bagging
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
    
    #With Bagging
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
    
    #Calculating the F1 Score
    varBag = classification_report(Y_Test,Y_Pred_Bagging,output_dict=True)['weighted avg']['f1-score']
    f1scoreBagging.append(varBag)
    
    var = classification_report(Y_Test,Y_Pred,output_dict=True)['weighted avg']['f1-score']
    f1score.append(var)
    
print("Mean F1-Score(Bagging) :", mean(f1scoreBagging),"\n")
print("Mean F1-Score(Without) :", mean(f1score))


# In[ ]:




