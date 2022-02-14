#!/usr/bin/env python
# coding: utf-8

# # Gradient Boosting
# 
# 

# ## Data
# 
# 

# In[1]:


from data_prep_credit import *


# ## Model

# In[2]:


from sklearn.ensemble import GradientBoostingClassifier


# In[3]:


clf = GradientBoostingClassifier(n_estimators=100, 
                                learning_rate=1.0,
                                max_depth=1, 
                                random_state=0).fit(X_train, y_train)

                                
y_pred = clf.fit(X_train, y_train).predict(X_test)


# In[4]:


clf.score(X_test, y_test)


# ### Confusion matrix

# In[5]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()


# ### Classification report

# In[6]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

