# Project 2


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from numpy.random import default_rng
from scipy import stats
```


```python
data = pd.read_csv('../data/nyse-data.csv')
data['date'] = pd.to_datetime(data['date'])
data
```
```python
apple_data = data[data['symbol'] == 'AAPL']
drop_date = '2014-06-09'
apple_data_pre_2015 = apple_data[apple_data['date'] < pd.to_datetime(drop_date)] # remove data > 2014-03-01

ax = apple_data.plot('date', 'open', xlabel='Date', title='NYSE Opening Price: AAPL', ylabel='Opening price (\$)', legend=False)
apple_data_pre_2015.plot('date', 'open', ax=ax, label=f'Pre {drop_date}');
```
```python
dprice = np.diff(apple_data_pre_2015['open'])
plt.plot(apple_data_pre_2015['date'].iloc[1:], dprice)
plt.title('AAPL Price Fluctuation')
plt.xlabel('Date')
plt.ylabel('Change in opening price (\$/day)');
```
```python
mean_dprice = np.mean(dprice)
std_dprice = np.std(dprice)
x = np.linspace(-40, 40)
price_pdf = stats.norm.pdf(x, loc = mean_dprice, scale = std_dprice)
```


```python
plt.hist(dprice, 100, density=True)
plt.plot(x, price_pdf)
plt.title('GOOGL changes in price over 4 years\n'+
         f'avg: \${mean_dprice:.2f} stdev: \${std_dprice:.2f}');
```
```python
rng = default_rng()
N_models = 100
dprice_model = rng.normal(size = (len(apple_data_pre_2015), N_models), loc = mean_dprice, scale = std_dprice)

plt.hist(dprice, 100, density=True, label = 'NYSE data')
plt.plot(x, price_pdf)
plt.hist(dprice_model[:, 0], 50, density = True, histtype='step', linewidth=3, label='model prediction 1')

start, stop = apple_data_pre_2015['date'].iloc[0], apple_data_pre_2015['date'].iloc[-1]
plt.title(f"AAPL changes in price from {start:%m-%d-%Y} to {stop:%m-%d-%Y}\n"+
         f'avg: \${mean_dprice:.2f} | stdev: \${std_dprice:.2f}')
plt.legend();
```
```python
rng = default_rng()
N_models = 100
dprice_model = rng.normal(size = (len(apple_data_pre_2015), N_models), loc = mean_dprice, scale = std_dprice)

price_model = np.cumsum(dprice_model, axis = 0) + apple_data_pre_2015['open'].values[0]
price_model_avg = np.mean(price_model, axis = 1)
price_model_std = np.std(price_model, axis = 1)

ax = plt.axes(xlabel='Date', ylabel='Opening Price (\$)', title=f'Random Walk Prediction ({N_models} models)')

ax.plot(apple_data_pre_2015['date'], price_model, alpha = 0.2, linewidth=2)
ax.plot(apple_data_pre_2015['date'], apple_data_pre_2015['open'], c = 'k', label = 'NYSE Data')
ax.plot(apple_data_pre_2015['date'], price_model_avg, c='r', label='Random Walk Average')
ax.fill_between(apple_data_pre_2015['date'], price_model_avg+price_model_std, price_model_avg-price_model_std, 
                color='r', alpha=.25, zorder=2, label='Random Walk Std Dev')
ax.legend();
```
```python

```
