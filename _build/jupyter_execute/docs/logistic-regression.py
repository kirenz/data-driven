#!/usr/bin/env python
# coding: utf-8

# # Logistic regression

# ## World happiness report

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
# - 1.1 Rename the variables in the DataFrame to 'Country', 'Happiness_Score', 'Economy', 'Family', 'Health' and 'Trust'. Check for missing values.    
# 
# - 1.2 Create a new categorical variable called `Happy`, where all countries with a Happiness_Score > 5.5 are labeled with 'Yes', otherwise 'No'.  
#     
# - 1.3 Delete the variable `Happiness_Score` and change the data types if necessary (categorical, float, integer...). 
# 
# - 1.4 Visualize the distributions of the numerical variables and display the distributions of the two groups (Happy: 'Yes' and 'No').  
# 
# - 1.5. Check for relationships with correlations.

# Logistic regression model:
# 
# - 2.a) Fit a logistic regression model with all predictor variables (response: `Happy`; predictors: `Economy`, `Family`, `Health` and `Trust`).  
# 
# - 2.b) Please explain wether you would recommend to exclude a predictor variable from your model (from task 2a)). Update your model if necessary.
# 
# - 2.c) Use your updated model and predict the probability that a country has "no happy" inhabitants. Classify this countries with label 'Yes' if the predicted probability exceeds:
# 
#     - c1): 0.4 (i.e. threshold = 0.4) 
#     - c2): 0.5 (i.e. threshold = 0.5)
#     - c3): 0.7 (i.e. threshold = 0.7). 
#     
# Otherwise classify the country as happy (with label 'No').
# 
# 
# - 2.d) Compute the confusion matrix for every threshold (c1), c2) and c3)) in order to determine how many observations were correctly or incorrectly classified. Furthermore, use the results from the confusion matrix and create the following variables: true positive; true negative; false positive and false negative. Use these variables to calculate the following measures: "Accuracy", Precision" (what proportion of positive identifications was actually correct?), "Recall" (what proportion of actual positives was identified correctly) and the F1 score (measure of a test's accuracy) for the thresholds in c1), c2) and c3). Which threshold would you recommend? 
# 
# Hints: **Precision** is defined as the number of true positives over the number of true positives plus the number of false positives. **Recall** is defined as the number of true positives over the number of true positives plus the number of false negatives. These two quantities are related to the **F1 score**, which is defined as the harmonic mean of precision and recall: $F1 = 2* ((Precision * Recall)/(Precision + Recall)).$
# 
# - 2.e) Fit the logistic regression model using a training data set. Compute the confusion matrix and accuracy for the held out data (test data size = 30%). Use a threshold of 0.5.

# ## Python setup

# In[1]:


import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import chi2_contingency, fisher_exact

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data preparation

# In[2]:


# Load the csv data files into pandas dataframes
PATH = 'https://raw.githubusercontent.com/kirenz/datasets/master/happy.csv' 
df = pd.read_csv(PATH)


# ## Tidying data

# ### Data inspection

# First of all, let's take a look at the data set.

# In[3]:


# show data set
df


# In[4]:


# Drop variables
df = df.drop('Unnamed: 0', axis=1)


# In[5]:


df.describe()


# ## 1.1 Rename and missing

# In[6]:


df.columns = ['Country', 'Happiness_Score', 'Economy', 'Family', 'Health', 'Trust']
df.head()


# Handle missing values

# In[7]:


print(df.isnull().sum())


# ## 1.2 Create flag

# In[8]:


df['Happy'] = np.where(df['Happiness_Score']>5.5, 'Yes', 'No')


# In[9]:


df


# In[10]:


df.Happy.value_counts()


# ## 1.3 Drop and data format

# In[11]:


df = df.drop('Happiness_Score', axis=1)


# In[12]:


df.info()


# In[13]:


# Change data types
df['Country'] = df['Country'].astype('category')
df['Happy'] = df['Happy'].astype('category')


# ## 1.4 Distribution

# In[14]:


sns.pairplot(hue="Happy", data=df);


# ## 1.5 Correlation

# In[15]:


# Inspect relationship between variables (correlation)
# Calculate correlation using the default method ( "pearson")
corr = df.corr()
# optimize aesthetics: generate mask for removing duplicate / unnecessary info
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap as indicator for correlations:
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Plot
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, square=True, annot_kws={"size": 12});


# ## 2a) Model with all predictors

# In[16]:


model = smf.glm(formula = 'Happy ~ Economy + Family + Health + Trust' , data=df, family=sm.families.Binomial()).fit()


# In[17]:


print(model.summary())


# Note that Statsmodels decoded the dependent variable as Happy "No" and Happy "Yes". Since "No" comes first, the model will predict the label "No".

# ## 2. b) Update Model

# In[18]:


# Define and fit logistic regression model
model_2 = smf.glm(formula = 'Happy ~ Economy + Family' , data=df, family=sm.families.Binomial()).fit()


# In[19]:


print(model_2.summary())


# ## 2c) Predict

# In[20]:


# Predict and join probabilty to original dataframe
df['Probability_no'] = model_2.predict()


# In[21]:


df


# In[22]:


# Use thresholds to discretize Probability
df['Threshold 0.4'] = np.where(df['Probability_no'] > 0.4, 'No', 'Yes')
df['Threshold 0.5'] = np.where(df['Probability_no'] > 0.5, 'No', 'Yes')
df['Threshold 0.6'] = np.where(df['Probability_no'] > 0.6, 'No', 'Yes')
df['Threshold 0.7'] = np.where(df['Probability_no'] > 0.7, 'No', 'Yes')

df


# ## 2. d) Confusion Matrix & Metrics

# In[23]:


def print_metrics(df, predicted):
    # Header
    print('-'*50)
    print(f'Metrics for: {predicted}\n')
    
    # Confusion Matrix
    y_actu = pd.Series(df['Happy'], name='Actual')
    y_pred = pd.Series(df[predicted], name='Predicted')
    df_conf = pd.crosstab(y_actu, y_pred)
    display(df_conf)
    
    # Confusion Matrix to variables:
    pop = df_conf.values.sum()
    tp = df_conf['Yes']['Yes']
    tn = df_conf['No']['No']
    fp = df_conf['Yes']['No']
    fn = df_conf['No']['Yes']
    
    # Metrics
    accuracy = (tp + tn) / pop
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {f1_score:.4f} \n')


# In[24]:


print_metrics(df, 'Threshold 0.4')
print_metrics(df, 'Threshold 0.5')
print_metrics(df, 'Threshold 0.6')
print_metrics(df, 'Threshold 0.7')


#   General examples to explain the concepts:
#   
#   - When we have a case where it is important to predict true positives correctly and ther is a cost associated with false positives, then we should use precision. 
#   
#   - The metric recall would be a good metric if we want to target as many true positive cases as possible and don't care a lot about false positives. 
# 
#   - If we want a balance between recall and precision, we should use the F-Score.

# ## 2. e) Use train test data 

# If we use train and test data, we need to change some of the steps above since we make use of a scikit learn library.

# In[25]:


# Encode happy = 0 and not happy = 1. Convert to float

y = pd.get_dummies(df['Happy'])
y = y['No'].astype('float')


# In[26]:


# Select features
X = df[['Economy', 'Family']]


# In[27]:


# Train test split
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=12)


# In[28]:


print("Training size:", len(train_X))
print("Test size:", len(test_X))


# In[29]:


# Train logistic regression model with training set
logit = sm.Logit(train_y, train_X).fit()


# In[30]:


print(logit.summary())


# In[31]:


# create empty dataframe
data = pd.DataFrame()

# include prediction from test data
data['Probability'] = logit.predict(test_X)


# In[32]:


# Calculate metrics
data['Happy'] = np.where(test_y == 1.0, 'No', 'Yes')  
data['Threshold 0.5'] = np.where(data['Probability'] > 0.5, 'No', 'Yes')  
data.head(7)


# In[33]:


print_metrics(data, 'Threshold 0.5')

