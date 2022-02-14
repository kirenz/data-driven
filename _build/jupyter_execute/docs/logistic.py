#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# We use a classification model to predict which customers will default on their credit card debt. 
# 
# ## Data
# 
# To learn more about the data and all of the data preparation steps, take a look at [this page](/docs/data-credit.ipynb). Here, we simply import a Python script which includes all of the necessary steps.

# In[1]:


from data_prep_credit import * 


# ## Model

# In[2]:


import sklearn.linear_model as skl_lm

clf = skl_lm.LogisticRegression()
y_pred = clf.fit(X_train, y_train).predict(X_test)


# In[3]:


# Return the mean accuracy on the given test data and labels:
clf.score(X_test, y_test)


# ### Confusion matrix

# In[4]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()


# ### Classification report

# In[5]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))


# ### Change threshold

# Use specific threshold

# In[6]:


pred_proba = clf.predict_proba(X_test)

df_ = pd.DataFrame({'y_test': y_test, 'y_pred': pred_proba[:,1] > .25})
cm = confusion_matrix(y_test, df_['y_pred'])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()


# ### Classification report

# In[7]:



print(classification_report(y_test, df_['y_pred']))

