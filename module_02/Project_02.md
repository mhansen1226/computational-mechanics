# Project 2


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-deep')
from numpy.random import default_rng
from scipy import stats
```


```python
data = pd.read_csv('../data/nyse-data.csv')
data['date'] = pd.to_datetime(data['date'])
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>symbol</th>
      <th>open</th>
      <th>close</th>
      <th>low</th>
      <th>high</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-05</td>
      <td>WLTW</td>
      <td>123.430000</td>
      <td>125.839996</td>
      <td>122.309998</td>
      <td>126.250000</td>
      <td>2163600.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-06</td>
      <td>WLTW</td>
      <td>125.239998</td>
      <td>119.980003</td>
      <td>119.940002</td>
      <td>125.540001</td>
      <td>2386400.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-07</td>
      <td>WLTW</td>
      <td>116.379997</td>
      <td>114.949997</td>
      <td>114.930000</td>
      <td>119.739998</td>
      <td>2489500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-08</td>
      <td>WLTW</td>
      <td>115.480003</td>
      <td>116.620003</td>
      <td>113.500000</td>
      <td>117.440002</td>
      <td>2006300.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-11</td>
      <td>WLTW</td>
      <td>117.010002</td>
      <td>114.970001</td>
      <td>114.089996</td>
      <td>117.330002</td>
      <td>1408600.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>851259</th>
      <td>2016-12-30</td>
      <td>ZBH</td>
      <td>103.309998</td>
      <td>103.199997</td>
      <td>102.849998</td>
      <td>103.930000</td>
      <td>973800.0</td>
    </tr>
    <tr>
      <th>851260</th>
      <td>2016-12-30</td>
      <td>ZION</td>
      <td>43.070000</td>
      <td>43.040001</td>
      <td>42.689999</td>
      <td>43.310001</td>
      <td>1938100.0</td>
    </tr>
    <tr>
      <th>851261</th>
      <td>2016-12-30</td>
      <td>ZTS</td>
      <td>53.639999</td>
      <td>53.529999</td>
      <td>53.270000</td>
      <td>53.740002</td>
      <td>1701200.0</td>
    </tr>
    <tr>
      <th>851262</th>
      <td>2016-12-30</td>
      <td>AIV</td>
      <td>44.730000</td>
      <td>45.450001</td>
      <td>44.410000</td>
      <td>45.590000</td>
      <td>1380900.0</td>
    </tr>
    <tr>
      <th>851263</th>
      <td>2016-12-30</td>
      <td>FTV</td>
      <td>54.200001</td>
      <td>53.630001</td>
      <td>53.389999</td>
      <td>54.480000</td>
      <td>705100.0</td>
    </tr>
  </tbody>
</table>
<p>851264 rows × 7 columns</p>
</div>




```python
apple_data = data[data['symbol'] == 'AAPL']
drop_date = '2014-06-09'
apple_data_pre_2015 = apple_data[apple_data['date'] < pd.to_datetime(drop_date)] # remove data > 2014-03-01

ax = apple_data.plot('date', 'open', xlabel='Date', ylabel='Opening price (\$)', legend=False)
apple_data_pre_2015.plot('date', 'open', ax=ax, label=f'Pre {drop_date}');
```


    
![png](Project_02_files/Project_02_3_0.png)
    



```python
dprice = np.diff(apple_data_pre_2015['open'])
plt.plot(apple_data_pre_2015['date'].iloc[1:], dprice)
plt.title('Price Fluctuation')
plt.xlabel('Date')
plt.ylabel('Change in opening price (\$/day)');
```


    
![png](Project_02_files/Project_02_4_0.png)
    



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


    
![png](Project_02_files/Project_02_6_0.png)
    



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


    
![png](Project_02_files/Project_02_7_0.png)
    



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


    
![png](Project_02_files/Project_02_8_0.png)
    



```python

```
