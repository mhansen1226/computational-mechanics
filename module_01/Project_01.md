---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Project 1: Heat Transfer in Forensic Science

```python
import numpy as np
import matplotlib.pyplot as plt
import math
```

From Newtons Law of Cooling: $\frac{dT}{dt} = -K(T - T_a)$


## Part 1


$\frac{dT}{dt} = \frac{T(t+\Delta t)-T(t)}{\Delta t}$

$K = \frac{T_0 - T_f}{\Delta t(T_0 - T_a)}$

```python
Ta = 65        # °F
T0, Tf = 85, 74 # °F
t0, tf = 0, 2  # 11am - 1pm
dt = tf - t0

K = (T0 - Tf) / (dt * (T0 - Ta))
print(f'{K = :.3f}')
```

## Part 2

```python
def cooling_constant(T0, Tf, Ta, dt):
    return (T0 - Tf) / (dt * (Tf - Ta))
```

## Part 3

### Solutions

Analytical Solution: $T(t)=T_a+(T(0)-T_a)e^{Kt}$

Numerical Approximation: $T(i+1) = T(i) - K * (T(i) - T_a)$

```python
dt = .5                         # hr
t = np.arange(t0, 10, dt)       # hr
T_an = T_num = np.zeros(len(t)) # °F
T_an[0] = T_num[0] = T0

T_an = Ta + (T0 - Ta)*np.exp(-K*t)

for i in range(1, len(t)):
    T_num[i] = T_num[i-1] - K*(T_num[i-1] - Ta)*(t[i]-t[i-1])

ax = plt.axes(xlabel='Time since 11:00am [hrs]', ylabel='Temperature [°F]', title=f'Cooling Curve:\nAnalytical vs. Numerical Solution ($\Delta t = ${dt} hrs)')
ax.plot(t, T_an, label='Analytical')
ax.plot(t, T_num, 'o-', label='Numerical')
ax.hlines(Ta, t[0], t[-1], linestyles='--', colors='black', label=f'$T_a$ = {Ta}°F')
ax.legend()
```

### As time approaches infinity

$T(t\rightarrow \infty)=T_a=65°F$


### Time of Death

```python
import datetime
dt = .001
t = np.arange(t0, 10, dt)   # hr
T_num = np.zeros(len(t))    # °F
T_num[0] = 96.8             # °F

T_an = Ta + (T0 - Ta)*np.exp(-K*t)
for i in range(1, len(t)):
    T_num[i] = T_num[i-1] - K*(T_num[i-1] - Ta)*(t[i]-t[i-1])

time_before_found = t0 - dt*(abs(T_num - T0)).argmin()
time_found = datetime.datetime(1,1,1,11)
time_of_death = time_found + datetime.timedelta(hours=time_before_found)
print(f'Time of death: {time_of_death:%H:%M %p}')
```
