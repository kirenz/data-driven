# Relationships

## Correlation with response

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

## Correlation between predictors

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
