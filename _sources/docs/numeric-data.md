# Numerical data

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
