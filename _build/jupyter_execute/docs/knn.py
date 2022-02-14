#!/usr/bin/env python
# coding: utf-8

# # K Nearest Neighbors
# 
# We use a classification model to predict which customers will default on their credit card debt. 

# ## Data
# 
# To learn more about the data and all of the data preparation steps, take a look at [this page](/docs/data-credit.ipynb). Here, we simply import a Python script which includes all of the necessary steps.

# In[1]:


from data_prep_credit import * 


# ## Model

# In[2]:


from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors=2)
y_pred = clf.fit(X_train, y_train).predict(X_test)


# ### Confusion matrix

# In[3]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()


# ### Classification report

# In[4]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, digits=3))

