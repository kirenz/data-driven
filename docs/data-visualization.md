# Data visualization

## Numerical data

For numerical data we take a look at the central tendency and distribution:

```Python
# summary of numerical attributes
df_train.describe().round(2).T
```

```Python
# histograms
df_train.hist(figsize=(20, 15));
```

Content of the following presentation: Scatterplots for paired data; dot plots and the mean; histograms and shape; Variance and standard deviation.

<br>

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQckk0QRSzfFAWcyxx8vO42WTBusDau6Su5NR7PD4cmBpqmI9Bq2cRYy_juPogZKWQGymUF9dhy7B9a/embed?start=false&loop=false&delayms=3000" frameborder="0" width="820" height="520" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

<br>


```{admonition} Resources
:class: tip
- [Download slides](https://docs.google.com/presentation/d/1hxGSzOcvwqBbmMsz0MCVRkuWP9e0ID3u3t0WNGpwItc/export/pdf)
- [Example in Google sheets](https://docs.google.com/spreadsheets/d/1xXhRBbKjqlglrUUdRKWhwG5XiMKQsbGqOagLara3V0k/edit#gid=0)
- Colab: [Dot plots and the mean](https://colab.research.google.com/github/kirenz/modern-statistics/blob/main/05-2-dot-plots-mean.ipynb)
- Colab: [Histograms](https://colab.research.google.com/github/kirenz/modern-statistics/blob/main/05-3-histograms.ipynb)
- Colab: [Case height](https://colab.research.google.com/github/kirenz/modern-statistics/blob/main/05-3-pairplot.ipynb)
- Reading: [Introduction to Modern Statistics (2021)](https://openintro-ims.netlify.app/explore-numerical.html)
```


(section:data:categorical)=
## Categorical data

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

## Numerical grouped by categorical

We can also investigate numerical data grouped by categorical data:

```Python
# median
for i in list_cat:
    print(df_train.groupby(i).median().round(2).T)
```

```Python
# mean
for i in list_cat:
    print(df_train.groupby(i).mean().round(2).T)
```

```Python
# standard deviation
for i in list_cat:
    print(df_train.groupby(i).std().round(2).T)
```

<br>

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRKNIZnYToIwcsrhiYqbX1gf5rvhPDXh7SACeg7YlokmKB85840iyG_zjbWrQIHwkhFjzROFous1noj/embed?start=false&loop=false&delayms=3000" frameborder="0" width="820" height="520" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

<br>

```{admonition} Resources
:class: tip
- [Download slides](https://docs.google.com/presentation/d/1GDhWpIMsyFWA2lZP9NERyncL84W3HuIkylendbVfq-g/export/pdf)
- Colab: [Comparisons across groups](https://colab.research.google.com/github/kirenz/modern-statistics/blob/main/04-6-comparisons-across-groups.ipynb)
- Reading: [Introduction to Modern Statistics (2021)](https://openintro-ims.netlify.app/explore-categorical.html#comparing-numerical-data-across-groups)
```

## Relationships

### Correlation with response

Detect the relationship between each predictor and the response:

```Python
sns.pairplot(data=df_train, y_vars=y_label, x_vars=features);
```

```Python
# pairplot with one categorical variable
sns.pairplot(data=df_train, y_vars=y_label, x_vars=features, hue="a_categorical_variable");
```

```Python
# inspect correlation
corr = df_train.corr()
corr_matrix[y_label].sort_values(ascending=False)
```

### Correlation between predictors

Investigate relationships between predictors to detect multicollinearity:

```Python
sns.pairplot(df_train);
```

```Python
# inspect correlation
corr = df_train.corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)

fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True,  
            square=True, annot_kws={"size": 12});
```
