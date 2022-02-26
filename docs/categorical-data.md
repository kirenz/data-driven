(section:data:categorical)=
# Categorical data

For categorical data we check the levels and their uniqueness:

```Python
df_train.describe(include="category").T 
```  

```Python
for i in list_cat:
    print(i, "\n", df_train[i].value_counts())
```  

```Python
for i in list_cat:
    print(df_train[i].value_counts().plot(kind='barh', title=i));
```  

If you have variables with many levels and are interested only in the top 10 values:

```Python
for i in cat_list:

    TOP_10 = df[i].value_counts().iloc[:10].index

    g = sns.catplot(y=i, 
            kind="count", 
            palette="ch:.25", 
            data=df,
            order = TOP_10)    
    
    plt.title(i)
    plt.show();
```


Content of the following presentation: Contingency tables and bar plots; bar plots with two variables; mosaic plots; row and column proportions; pie charts; waffle charts

<br>

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRyIy9mif940HDWeF3qkPwfcO68n-KXwbmq2B50eUcovtbMcO3LdvJYKnqwJ3JudWG9Q2l0qKwH9pyf/embed?start=false&loop=false&delayms=3000" frameborder="0" width="820" height="520" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

<br>

```{admonition} Resources
:class: tip
- [Download slides](https://docs.google.com/presentation/d/1s5-_4lGxJERlPQxpb3adG99iH9b5hiyarK2fGBwHhIY/export/pdf)
- Colab: [Contingency table and bar plot](https://colab.research.google.com/github/kirenz/modern-statistics/blob/main/04-1-contingency-table-bar-plot.ipynb)
- Colab: [Two categorical variables](https://colab.research.google.com/github/kirenz/modern-statistics/blob/main/04-2-two-categorical-variables.ipynb)
- Colab: [Row and column proportions ](https://colab.research.google.com/github/kirenz/modern-statistics/blob/main/04-3-row-column-proportions.ipynb)
- Colab: [Pie charts](https://colab.research.google.com/github/kirenz/modern-statistics/blob/main/04-4-pie-charts.ipynb)
- Colab: [Waffle charts](https://colab.research.google.com/github/kirenz/modern-statistics/blob/main/04-5-waffle-charts.ipynb)
- Reading: [Introduction to Modern Statistics (2021)](https://openintro-ims.netlify.app/explore-categorical.html#contingency-tables-and-bar-plots)
```
