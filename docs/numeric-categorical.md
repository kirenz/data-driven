# Numerical grouped by categorical

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
