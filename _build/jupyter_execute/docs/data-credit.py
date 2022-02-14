#!/usr/bin/env python
# coding: utf-8

# (section-data)=
# # Credit data

# The credit data is a simulated data set containing information on ten thousand customers (taken from {cite:t}`James2021`). The aim here is to use a classification model to predict which customers will default on their credit card debt (i.e., failure to repay a debt):
# 
# - default: A categorical variable with levels No and Yes indicating whether the customer defaulted on their debt 
# - student: A categorical variable with levels No and Yes indicating whether the customer is a student
# - balance: The average balance that the customer has remaining on their credit card after making their monthly payment
# - income: Income of customer

# ## Import data

# In[1]:


import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/kirenz/classification/main/_static/data/Default.csv')


# ## Inspect data

# In[2]:


df


# In[3]:


df.info()


# In[4]:


# check for missing values
print(df.isnull().sum())


# ##  Data preparation

# ### Categorical data
# 
# First, we convert categorical data into indicator variables: 

# In[5]:


dummies = pd.get_dummies(df[['default', 'student']], drop_first=True, dtype=float)
dummies.head(3)


# In[6]:


# combine data and drop original categorical variables
df = pd.concat([df, dummies], axis=1).drop(columns = ['default', 'student'])
df.head(3)


# ### Label and features 

# Next, we create our y label and features:

# In[7]:


y = df.default_Yes
X = df.drop(columns = 'default_Yes')


# ### Train test split

# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)


# ## Data exploration

# Create data for exploratory data analysis.

# In[9]:


train_dataset = pd.DataFrame(X_train.copy())
train_dataset['default_Yes'] = pd.DataFrame(y_train)


# In[10]:


import seaborn as sns

sns.pairplot(train_dataset, hue='default_Yes');

