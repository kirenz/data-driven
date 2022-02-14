#!/usr/bin/env python
# coding: utf-8

# # Classification with logistic regression

# In this problem set we analyse the relationship between online ads and purchase behavior. In particular, we want to classify which online users are likely to purchase a certain product after being exposed to an online ad.  
# 
# ## Data preparation

# In[1]:


import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/kirenz/datasets/master/purchase.csv")
df


# In[2]:


df.info()


# In[3]:


# make dummy variable
df['male'] = pd.get_dummies(df['Gender'], drop_first = True)
# drop irrelevant columns
df.drop(columns= ['Unnamed: 0', 'User ID', 'Gender'], inplace = True)


# In[4]:


# inspect outcome variable
df['Purchased'].value_counts()


# In[5]:


# prepara data for scikit learn 
X = df.drop(columns=['Purchased'])
y = df.Purchased


# In[6]:


# make data split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)


# In[7]:


# create new training dataset for data exploration
train_dataset = pd.DataFrame(X_train).copy()
train_dataset['Purchased'] = pd.DataFrame(y_train)

train_dataset


# ## Exploratory data analysis (EDA)

# In[8]:


train_dataset.groupby(by=['Purchased']).describe().T


# Purchasers are (on average) _______ and earn a __________ estimated salary than non-purchasers. 
# 
# Visualization of differences:

# In[43]:


import seaborn as sns

sns.pairplot(hue='Purchased', kind="reg", diag_kind="kde", data=train_dataset);


# Inspect (linear) relationships between variables with correlation (pearson's correlation coefficient)

# In[41]:


import numpy as np

# Calculate correlation using the default method ( "pearson")
corr = train_dataset.corr()
# optimize aesthetics: generate mask for removing duplicate / unnecessary info
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap as indicator for correlations:
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Plot
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, square=True, annot_kws={"size": 12});


# In[9]:


sns.kdeplot(hue="Purchased", x='Age', data=train_dataset);


# Purchasers seem to be _________ than non-purchaser.

# In[10]:


sns.boxplot(x="male", y="Age", hue="Purchased", data=train_dataset);


# There are __________ differences regarding gender.

# In[11]:


sns.kdeplot(hue="Purchased", x='EstimatedSalary', data=train_dataset); 


# Purchaser earn a ______________ estimated salary.

# In[12]:


sns.boxplot(x="male", y="EstimatedSalary", hue="Purchased", data=train_dataset);


# Insight: there are ___________ differences between males and females (regarding purchase behavior, age and estimated salary)

# ## Model
# 
# Next, we will fit a logistic regression model with a [L2 regularization (ridge regression)](https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization). In particular, we use an estimator that has built-in cross-validation capabilities to automatically select the best hyper-parameter for our L2 regularization (see the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV)).
# 
# We only use our most promising predictor variables `Age` and `EstimatedSalary` for our model.

# In[30]:


# only use meaningful predictors
features_model = ['Age', 'EstimatedSalary']

X_train = X_train[features_model] 
X_test = X_test[features_model]


# In[31]:


import sklearn.linear_model as skl_lm

# model
clf = skl_lm.LogisticRegressionCV(penalty='l2')
# prediction
y_pred = clf.fit(X_train, y_train).predict(X_test)


# ## Classification metrics

# In[32]:


# Return the mean accuracy on the given test data and labels:
clf.score(X_test, y_test)


# In[33]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()


# In[34]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))


# ``macro``: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# 
# ``weighted``: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters 'macro' to account for label imbalance.
# 
# Note that recall is also sometimes called sensitivity or true positive rate.

# * High scores for both *precision* and *recall* show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).
# 
# * The importance of precision vs recall depends on the use case at hand (and the costs associated with missclassification). 
# 
# * A system with *high recall* but *low precision* returns many results, but most of its predicted labels are incorrect when compared to the training labels. 
# 
# * A system with *high precision* but *low recall* is just the opposite, returning very few results, but most of its predicted labels are correct when compared to the training labels. 
#   
# * An ideal system with high precision and high recall will return many results, with most results labeled correctly. 

# The unweighted recall of our model is _____  
# 
# The unweighted precision of our model is _____  

# ## Predictions
# 
# The `predict()` function can be used to predict the probability that a customer will buy, given values of the predictors. The `type="response"` option tells R to output probabilities of the form P(Y = 1|X).

# In[ ]:


glm_probs <-predict(glm_fits, type ="response")
round(glm_probs[1:10],2)


# We can use the `contrasts()` function to check wether the above values actually correspond to the probability of a purchase. Below the `contrasts()` function indicates that R has created a dummy variable with a 1 for purchase.

# In[ ]:


contrasts(purchase$Purchased)


# ## Thresholds
# 
# Logistic regression returns a probability. You can use the returned probability "as is" (for example, the probability that the user will buy the product is 0.8) or convert the returned probability to a binary value (for example, this user will buy the product, therefore we label him as "Yes").
# 
# A logistic regression model that returns 0.9 for a particular customer is predicting that it is very likely that the customer will buy the product. In order to map a logistic regression value to a binary category (e.g., "Yes" or "No"), you must define a **classification threshold** (also called the decision threshold). A value above that threshold indicates "Yes", the customer will buy the product; a value below indicates "No", the customer will not buy the product. 
# 
# Notice that the optimal classification threshold is problem-dependent and therefore a value that you must tune (see [Google developers](https://developers.google.com/machine-learning/crash-course/classification/thresholding)).
# 
# We use three different thresholds. Which threshold would you recommend?

# In[ ]:


pred_proba = clf.predict_proba(X_test)


# ### Threshold  0.4

# In[44]:


df_ = pd.DataFrame({'y_test': y_test, 'y_pred': pred_proba[:,1] > .4})
cm = confusion_matrix(y_test, df_['y_pred'])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()

print(classification_report(y_test, df_['y_pred']))


# ### Threshold 0.5

# In[45]:


df_ = pd.DataFrame({'y_test': y_test, 'y_pred': pred_proba[:,1] > .5})
cm = confusion_matrix(y_test, df_['y_pred'])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()

print(classification_report(y_test, df_['y_pred']))


# ### Threshold 0.7

# In[46]:


df_ = pd.DataFrame({'y_test': y_test, 'y_pred': pred_proba[:,1] > .7})
cm = confusion_matrix(y_test, df_['y_pred'])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()

print(classification_report(y_test, df_['y_pred']))

