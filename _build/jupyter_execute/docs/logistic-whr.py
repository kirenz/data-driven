#!/usr/bin/env python
# coding: utf-8

# # World happiness report

# 
# The World Happiness Report ranks 156 countries by their happiness levels (`Happiness.Score`). The rankings come from the Gallup World Poll and are based on answers to the main life evaluation question asked in the poll. This is called the Cantril ladder: it asks respondents to think of a ladder, with the best possible life for them being a 10, and the worst possible life being a 0. They are then asked to rate their own current lives on that 0 to 10 scale. The rankings are from nationally representative samples, for the years 2013-2015. They are based entirely on the survey scores, using the Gallup weights to make the estimates representative. 
# 
# The other variables in the dataset show the estimated extent to which each of the factors contribute to making life evaluations higher in each country than they are in Dystopia, a hypothetical country that has values equal to the world’s lowest national averages for each of the factors (for more information about the data, visit this [FAQ-site](https://s3.amazonaws.com/happiness-report/2016/FAQ_2016.pdf) or the [World Happiness Report-site](http://worldhappiness.report/ed/2016/)).
# 
# Data Source: 
# 
# *Helliwell, J., Layard, R., & Sachs, J. (2016). World Happiness Report 2016, Update (Vol. I).
# New York: Sustainable Development Solutions Network.*

# 
# ## Task Description
# 
# In this assignment, we analyse the relationship between the country specific happiness and some predictor variables. In particular, we want to classify which countries are likely to be "happy". 
# 
# 
# Data preparation:
# 
# - 1.0 Import the data and perform a quick data inspection.  
# 
# - 1.1 Drop the variable 'Unnamed: 0' and rename the variables in the DataFrame to 'Country', 'Happiness_Score', 'Economy', 'Family', 'Health' and 'Trust'. 
# 
# - 1.2 Create a new categorical variable called `Happy`, where all countries with a Happiness_Score > 5.5 are labeled with 1, otherwise 0.  
#     
# - 1.3 Delete the variable `Happiness_Score` 
#   
# - 1.4 Use scikit-learn to make the train test split (`X_train`, `X_test`, `y_train`, `y_test`) and create a pandas data exploration set from your training data (`df_train`). 
#   
# - 1.5 Perform exploratory data analysis to find differences in the distributions of the two groups (Happy: `1` and `0`).  
# 
# - 1.6. Check for relationships with correlations (use pairwise correlations and variance inflation factor).

# Logistic regression model:
# 
# - 2.0 Fit a logistic regression model with the following predictor variables (response: `Happy`; predictors: `Family`, `Health` and `Trust`) on the pandas training data (df_train).
# 
# - 2.1 Please explain wether you would recommend to exclude a predictor variable from your model (from task 2a)). Update your model if necessary.
# 
# - 2.2 Use your updated model and predict the probability that a country has "happy" inhabitants (use df_train) based on 3 different thresholds. In particular, classify countries with label `1` if the predicted probability exceeds the thresholds stated below (otherwise classify the country as happy (with `0`)) :
# 
#   - 0.4 (i.e. threshold = 0.4) 
#   - 0.5 (i.e. threshold = 0.5)
#   - 0.7 (i.e. threshold = 0.7)
# 
# - 2.3 Compute the classification report (`from sklearn.metrics import classification_report`) in order to determine how many observations were correctly or incorrectly classified. Which threshold would you recommend? 
# 
# - 2.4 Use the test data to evaluate your model with a threshold of 0.5.

# ## 1.0 Import data

# In[281]:


import pandas as pd

# Load the csv data files into pandas dataframes
PATH = 'https://raw.githubusercontent.com/kirenz/datasets/master/happy.csv' 
df = pd.read_csv(PATH)


# First of all, let's take a look at the data set.

# In[282]:


# show data set
df


# In[283]:


df.info()


# ## 1.1 Drop and rename

# In[284]:


# Drop variables we don't need
df = df.drop('Unnamed: 0', axis=1)


# In[285]:


df.columns = ['Country', 'Happiness_Score', 'Economy', 'Family', 'Health', 'Trust']
df.head()


# ## 1.2 Create flag

# In[286]:


import numpy as np

df['Happy'] = np.where(df['Happiness_Score']>5.5, 1, 0)


# In[287]:


df


# In[288]:


df.Happy.value_counts()


# ## 1.3 Drop feature

# In[289]:


df = df.drop('Happiness_Score', axis=1)


# In[290]:


df.info()


# ## 1.4 Data splitting

# In[291]:


from sklearn.model_selection import train_test_split

X = df.drop(columns = ['Country', 'Happy'])
y = df['Happy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)


# In[292]:


# Make data exploration set
df_train = pd.DataFrame(X_train.copy())
df_train = df_train.join(pd.DataFrame(y_train))

df_train


# ## 1.5 EDA 

# In[293]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

sns.pairplot(hue="Happy", data=df_train);


# ## 1.6 Correlation
# 
# Inspect pairwise relationship between variables (correlation):

# In[294]:


# Calculate correlation using the default method ( "pearson")
corr = df_train.corr()
# optimize aesthetics: generate mask for removing duplicate / unnecessary info
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap as indicator for correlations:
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Plot
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, square=True, annot_kws={"size": 12});


# We see that some of the features are correlated. In particular, Economy and Health are highly correlated. This means we should not include both Economy and Health as predictors in our model. 
# 
# Let`s also check the variance inflation factor for multicollinearity (which should not exceed 5):  

# In[295]:


from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# choose features and add constant
features = add_constant(df_train)
# create empty DataFrame
vif = pd.DataFrame()
# calculate vif
vif["VIF Factor"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
# add feature names
vif["Feature"] = features.columns

vif.round(2)


# We observe that Economy has a fairly high VIF of almost 5. Note that since we check for multicollinearity, we don't catch the high pairwise correlation between Economy and Health. As a matter of fact, depending on your data, you could also have low pairwise correlations, but have high VIF's.
# 
# Based on the findings of our correlation analysis, we will drop the feature Economy from our model.

# ## 2.0 First model 

# In this example, we use the statsmodel's formula api to train our model. Therefore, we use our pandas `df_train` data. 

# In[296]:


import statsmodels.formula.api as smf

model = smf.glm(formula = 'Happy ~ Health + Family + Trust' , data=df_train, family=sm.families.Binomial()).fit()


# In[297]:


print(model.summary())


# In this case, the model will predict the label "1".

# ## 2.1 Update Model

# In[298]:


# Define and fit logistic regression model
model_2 = smf.glm(formula = 'Happy ~ Health + Family' , data=df_train, family=sm.families.Binomial()).fit()


# In[299]:


print(model_2.summary())


# ## 2.2 Thresholds 

# In[300]:


# Predict and join probabilty to original dataframe
df_train['y_score'] = model_2.predict()


# In[301]:


df_train


# In[302]:


# Use thresholds to discretize Probability
df_train['thresh_04'] = np.where(df_train['y_score'] > 0.4, 1, 0)
df_train['thresh_05'] = np.where(df_train['y_score'] > 0.5, 1, 0)
df_train['thresh_07'] = np.where(df_train['y_score'] > 0.7, 1, 0)

df_train


# ## 2.3 Classification report

# Example of confusion matrix (not necessary)

# In[303]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(df_train['Happy'], df_train['thresh_04'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Happy', 'Not Happy'])

disp.plot();


# Now we use scikit learn`s classification report:

# In[304]:


from sklearn.metrics import classification_report

target_names = ['Happy', 'Not Happy']
print(classification_report(df_train['Happy'], df_train['thresh_04'], target_names=target_names))


# Show all classification reports with a for loop:

# In[305]:


list = ['thresh_04', 'thresh_05', 'thresh_07']

for i in list:
     print("Threshold:", i)
     print(classification_report(df_train['Happy'], df_train[i], target_names=target_names))


#   General examples to explain the concepts:
#   
#   - When we have a case where it is important to predict true positives correctly and there is a cost associated with false positives, then we should use precision (typically we use the `macro avg`). 
#   
#   - The metric recall (the `macro avg`) would be a good metric if we want to target as many true positive cases as possible and don't care a lot about false positives. 
# 
#   - If we want a balance between recall and precision, we should use the F1-Score (again the `macro avg`).

# ## 2.4 Use test data 
# 
# Note that we don`t need to create a pandas dataframe.

# In[306]:


# Predict test data
y_score_test = model_2.predict(X_test)

thresh_05_test = np.where(y_score_test > 0.5, 1, 0)

print(classification_report(y_test, thresh_05_test, target_names=target_names))

