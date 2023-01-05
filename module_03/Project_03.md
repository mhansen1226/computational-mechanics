# Project 3

![Initial condition of firework with FBD and sum of momentum](../images/firework.png)

$$m\frac{dv}{dt} = u\frac{dm}{dt} -mg - cv^2~~~~~~~~(1)$$

$$m\frac{dv}{dt} = u\frac{dm}{dt}~~~~~(2.a)$$

$$\frac{m_{f}}{m_{0}}=e^{-\Delta v / u},~~~~~(2.b)$$

### Imports


```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

### Initial Conditions


```python
y0 = 0                  # initial y-position        (m)
v0 = 0                  # initial velocity          (m/s)
m0, mf = 0.25, 0.05     # initial/final mass        (kg)
dm = 0.05               # constant mass change      (kg/s)
u = 250                 # average propellant speed  (m/s)
H = 300                 # desired height            (m)
```

## Simple Rocket Model

$state = \left[\begin{array}{c} y \\ v \\ m \end{array}\right]$

$\frac{d~state}{dt} = f(state)$

$\left[\begin{array}{c} v\\a\\ \frac{dm}{dt} \end{array}\right] = \left[\begin{array}{c} v\\ \frac{u}{m}\frac{dm}{dt} \\ -\frac{dm}{dt} \end{array}\right]$


```python
def simplerocket(state, dmdt=dm, u=u):
    '''Computes the right-hand side of the differential equation for the acceleration of a rocket, without drag or gravity, in SI units.
    
    Arguments:
    ----------    
    state : array of three dependent variables [y v m]^T
    dmdt  : mass rate change of rocket      (default is 0.05 kg/s)
    u     : speed of propellent expelled    (default is 250 m/s)
    
    Returns:
    --------
    derivs : array of three derivatives [v (u/m*dmdt-g-c/mv^2) dmdt]^T
    '''

    y, v, m = state
    return np.array([v, u/m*dmdt, -dmdt])
```

## Realistic Rocket Model

$\frac{d~state}{dt} = f(state)$

$\left[\begin{array}{c} v\\a\\ \frac{dm}{dt} \end{array}\right] = 
\left[\begin{array}{c} v\\ \frac{u}{m}\frac{dm}{dt}-g-\frac{c}{m}v^2 \\ -\frac{dm}{dt} \end{array}\right]$


```python
def rocket(state, dmdt=dm, u=u, c=0.18e-3):
    '''Computes the right-hand side of the differential equation for the acceleration of a rocket, with drag, in SI units.
    
    Arguments:
    ----------    
    state : array of three dependent variables [y v m]^T
    dmdt  : mass rate change of rocket      (default set to 0.05 kg/s)
    u     : speed of propellent expelled    (default set to 250 m/s)
    c     : drag constant for a rocket      (default set to 0.18e-3 kg/m)
    
    Returns:
    --------
    derivs : array of three derivatives [v (u/m*dmdt-g-c/mv^2) dmdt]^T
    '''
    g = 9.81
    
    y, v, m = state
    return np.array([v, (u/m*dmdt - g - c/m*v**2), -dmdt])
```

## Numerical Solution Step Functions


```python
def heun_step(state,rhs,dt,etol=0.000001,maxiters = 100):
    '''Update a state to the next time increment using the implicit Heun's method.
    
    Arguments
    ---------
    state : array of dependent variables
    rhs   : function that computes the RHS of the DiffEq
    dt    : float, time increment
    etol  : tolerance in error for each time step corrector
    maxiters: maximum number of iterations each time step can take
    
    Returns
    -------
    next_state : array, updated after one time increment'''
    e=1
    eps=np.finfo('float64').eps
    next_state = state + rhs(state)*dt

    for n in range(0,maxiters):
        next_state_old = next_state
        next_state = state + (rhs(state)+rhs(next_state))/2*dt
        e=np.sum(np.abs(next_state-next_state_old)/np.abs(next_state+eps))
        if e<etol:
            break
    return next_state
```


```python
def rk2_step(state, rhs, dt):
    '''Update a state to the next time increment using modified Euler's method.
    
    Arguments
    ---------
    state : array of dependent variables
    rhs   : function that computes the RHS of the DiffEq
    dt    : float, time increment
    
    Returns
    -------
    next_state : array, updated after one time increment'''
    
    mid_state = state + rhs(state) * dt*0.5    
    next_state = state + rhs(mid_state)*dt
 
    return next_state
```


```python
def euler_step(state, rhs, dt, **kwargs):
    '''Uses Euler's method to update a state to the next one. 
    
    Arguments:
    ----------
    state : array of two dependent variables [y v]^T
    rhs   : function that computes the right hand side of the differential equation.
    dt    : float, time increment. 
    
    Returns:
    --------
    next_state : array, updated state after one time increment.       
    '''
    
    next_state = state + rhs(state, **kwargs) * dt
    return next_state
```

## Solver Class


```python
class RocketSolver:
    '''Class that saves initial conditions, solves, and plots the rocket equations
    
    Attributes:
    -----------
    func   : the function used to model rocket flight
    y0     : initial y position
    v0     : initial velocity
    m0     : initial mass
    t      : time array for the latest solution
    model  : [y, v, m] array for the latest solution
    solver : solver function used to solve the latest solution
    
    Methods:
    --------
    initialize_model(N) -> arr:
        - initializes the array for numerical solutions
    
    solve_model(solver_func, N=100) -> t, model: 
        - uses solver_func to numerically solve rocket equations for N steps
    
    plot(yvm=0, t=None, model=None, ax=None) -> ax:
        - plot model[:, yvm] vs t on ax
    
    plot_all(self, t=None, model=None, axes=None) -> axes: 
        - makes a subolot of the y-position, velocity, and mass solutions against t on axes
    
    plot_against_analytical(self, t=None, model=None, axes=None) -> axes:
        - plot dv/u vs m0/mf and the tsiolkovsky equation on the first axes, error on the second
    '''
    
    def __init__(self, func, y0=y0, v0=v0, m0=m0, mf=mf, dm=dm):
        '''Initialize the RocketSolver with defaults listed in the Initial Conditions section '''
        self.func = func
        self.y0 = y0
        self.v0 = v0
        self.m0 = m0
        self.mf = mf
        self.dm = dm
    
    def initialize_model(self, N):
        '''Initializes an array for numerical solutions with the initial conditions
        
        Arguments:
        ----------
        N  : number of steps
        IC : initial conditions
        
        Returns:
        --------
        arr: array with initial conditions in index 0
        '''
        
        arr = np.zeros((N, 3))
        for i, IC in enumerate((self.y0, self.v0, self.m0)):
            arr[0, i] = IC
        return arr

    def solve_model(self, solver_func, N=100):
        '''Use solver_func to numerically solve rocket equations for N steps. Utilizes the 
        initialize_model() function to initialize the array to solve
        
        Arguments:
        ----------
        solver_func : Function to step through numerical solution (e.g. euler_step, rk2_step, heun_step)
        N           : number of steps (default is 100)
        
        Returns:
        --------
        t     : equally spaced time array derived from the mass steps
        model : array containing data for position, velocity, and mass as described in the model functions
        '''
        
        t = np.linspace(0, (self.m0-self.mf)/self.dm, N)
        dt = t[1] - t[0]
        model = self.initialize_model(N)
        for i in range(N-1):
            model[i+1] = solver_func(model[i], self.func, dt)
        self.t = t
        self.model = model
        self.solver = solver_func
            
        return t, model
    
    def plot(self, yvm=0, t=None, model=None, ax=None):
        '''plot t vs one variable of model (chosen with yvm) on ax
        
        Arguments:
        ----------
        yvm   : choice of plot 0, 1, or 2 for position, velocity, or mass plot respectively (else raises ValueError)
        t     : time array, if omitted uses self.t (the time array of the most recent solve_model() call)
        model : model array, if omitted uses self.model (the model array of the most recent solve_model() call)
        ax    : axes to plot on, if omitted creates an axes to plot on
        
        Returns:
        --------
        ax : the axes used for the plot
        '''
        
        if t is None:
            t = self.t
        if model is None:
            model = self.model
        if ax is None:
            ax = plt.axes()
        
        if yvm == 0: # position plot
            ax.set(title='Position vs Time', ylabel='y-position (m)')
        elif yvm == 1: # velocity plot
            ax.set(title='Velocity vs Time', ylabel='velocity (m/s)')
        elif yvm == 2: # mass plot
            ax.set(title='Mass vs Time', ylabel='mass (kg)')
        else:
            raise ValueError('Error: yvm must be 0, 1, or 2 for position, velocity, or mass plot respectively')
        
        ax.set_xlabel('time (s)')
        
        ax.plot(t, model[:, yvm], label=f'{self.solver.__name__}({self.func.__name__})')
        ax.legend()
        return ax
    
    def plot_all(self, t=None, model=None, axes=None):
        '''Makes a subolot of the y-position, velocity, and mass solutions against t on axes
        Uses plot() method
        
        Arguments:
        ----------
        t     : time array, if omitted uses self.t (the time array of the most recent solve_model() call)
        model : model array, if omitted uses self.model (the model array of the most recent solve_model() call)
        axes  : axes to plot on, if omitted creates axes to plot on
        
        Returns:
        --------
        axes : the axes used for the plot
        '''
        
        if t is None:
            t = self.t
        if model is None:
            model = self.model
        if axes is None:
            fig, axes = plt.subplots(1, 3)
            fig.set_figwidth(20)
        
        for i, ax in enumerate(axes):
            self.plot(i, t, model, ax)
        return axes
    
    def plot_against_analytical(self, t=None, model=None, axes=None):
        '''Plots dv/u vs m0/mf and the tsiolkovsky equation on the first axes, error on the second
        
        Arguments:
        ----------
        t     : time array, if omitted uses self.t (the time array of the most recent solve_model() call)
        model : model array, if omitted uses self.model (the model array of the most recent solve_model() call)
        axes  : axes to plot on, if omitted creates axes to plot on
        
        Returns:
        --------
        axes : the axes used for the plot
        '''
        
        if t is None:
            t = self.t
        if model is None:
            model = self.model
        if axes is None:
            fig, (eq_ax, err_ax) = plt.subplots(1, 2)
            fig.set_figwidth(12)
            eq_ax.set(xlabel='$m_0/m_f$', ylabel='$\Delta v/u$', title=r'Tsiolkovsky: $\frac{\Delta v}{u}=\ln(\frac{m_0}{m_f})$')
            err_ax.set(xlabel='$m_0/m_f$', ylabel='error', title=r'Error: $|u\ln(\frac{m_0}{m_f}) - \Delta v|$')
            fig.suptitle('Analytical vs Numerical')
        y, v, m = model.T
        
        # tsiolkovsky equation
        eq_ax.plot(m0/m, np.log(m0/m)*u, label='tsiolkovsky equation')
        
        # numerical solution
        eq_ax.plot(m0/m, v, label=f'{self.solver.__name__}({self.func.__name__})')
        eq_ax.legend()
        
        # error
        error = np.abs(np.log(m0/m)*u - v)
        err_ax.plot(m0/m, error)
        
        return eq_ax, err_ax
```

## Root Finding Functions


```python
def incsearch(func,xmin,xmax,ns=50):
    '''Incremental search root locator: finds brackets of x that contain sign changes of a function on the interval [xmin, xmax]
    
    Arguments:
    ---------
    func       : name of function
    xmin, xmax : endpoints of interval
    ns         : number of subintervals (default set to 50)
    
    Returns:
    ---------
    xb(k,1) is the lower bound of the kth sign change
    xb(k,2) is the upper bound of the kth sign change
    If no brackets found, xb = [].'''
    
    x = np.linspace(xmin,xmax,ns)
    f = [func(i) for i in x]
    sign_f = np.sign(f)
    delta_sign_f = sign_f[1:]-sign_f[0:-1]
    i_zeros = np.nonzero(delta_sign_f!=0)
    nb = len(i_zeros[0])
    xb = np.block([[ x[i_zeros[0]]],[x[i_zeros[0]+1]]] )

    if nb==0:
      print(f'>> No brackets found: check interval or increase ns ({ns = })')
    else:
      print(f'>> Number of brackets: {nb}')
    return xb
```


```python
def mod_secant(func,dx,x0,es=0.000001,maxit=50, **kwargs):
    '''Uses modified secant method to find the root of func
    
    Arguments:
    ----------
    func     : function to find roots of
    dx       : perturbation fraction
    xr       : initial guess
    es       : desired relative error                (default = 0.0001)
    maxit    : maximum allowable iterations          (default = 50)
    **kwargs : additional parameters to pass to func
    
    Returns:
    --------
    root : real root
    fx   : func evaluated at root
    ea   : approximate relative error 
    iter : number of iterations'''

    itrs = 0
    xr = x0
    for itrs in range(0,maxit):
        xrold = xr
        dfunc = (func(xr+dx) - func(xr))/dx
        xr = xr - func(xr)/dfunc
        if xr != 0:
            ea = abs((xr - xrold)/xr) * 100
        else:
            ea = abs(xr - xrold) * 100
        if ea <= es:
            break
    return xr,[func(xr), ea, itrs]
```


```python
def f_dm(dmdt, m0=m0, c=0.18e-3, u=u, height_desired=H):
    '''Define a function f_dm(dmdt) that returns height_desired-height_predicted[-1]
    here, the time span is based upon the value of dmdt
    
    Arguments:
    ---------
    dmdt           : the unknown mass change rate
    m0             : the known initial mass (default set to m0 as defined in Initial Conditions)
    c              : the known drag in kg/m (default set to 0.18e-3)
    u              : the known speed of the propellent (default set to u as defined in Initial Conditions)
    height_desired : the desired height (default set to H as defined in Initial Conditions)
    Returns:
    --------
    error : the difference between height_desired and height_predicted[-1] when f_dm(dmdt) = 0, the correct mass change rate was chosen
    '''
    rocket_solver = RocketSolver(rocket, dm=dmdt)
    t, model = rocket_solver.solve_model(heun_step)
    height_predicted , v, m = model.T
    error = height_desired-height_predicted[-1]
    return error
```

## Part 1: Simple Rocket Analysis


```python
simple_rocket_solver = RocketSolver(simplerocket)
```


```python
simple_rocket_solver.solve_model(euler_step)
axes = simple_rocket_solver.plot_all()
simple_rocket_solver.solve_model(heun_step)
simple_rocket_solver.plot_all(axes=axes);
```


    
![png](Project_03_files/Project_03_25_0.png)
    



```python
simple_rocket_solver.plot_against_analytical();
```


    
![png](Project_03_files/Project_03_26_0.png)
    


## Part 2: More Realistic Rocket Model


```python
rocket_solver = RocketSolver(rocket)
```


```python
rocket_solver.solve_model(euler_step)
axes = rocket_solver.plot_all()
rocket_solver.solve_model(heun_step)
rocket_solver.plot_all(axes=axes);
```


    
![png](Project_03_files/Project_03_29_0.png)
    



```python
rocket_solver.plot_against_analytical();
```


    
![png](Project_03_files/Project_03_30_0.png)
    


## Part 3: Finding Mass Change Rate to Reach 300 m


```python
iters = 10
dm_min, dm_max = 0.05, 0.4
print(f'Running incremental search from {dm_min} to {dm_max} for {iters} iterations ...')
x = incsearch(f_dm, dm_min, dm_max, iters)
print(f'>> Root between {x[0,0]:.4f} - {x[1, 0]:.4f}')
print(f'\nRunning modified secant with initial guess {x[0,0]:.4f} ...')
dm_300m, _ = mod_secant(f_dm, .00001, x[0,0])
print(f'>> The correct mass change rate: {dm_300m:.4e} kg/s')
```

    Running incremental search from 0.05 to 0.4 for 10 iterations ...
    >> Number of brackets: 1
    >> Root between 0.0500 - 0.0889
    
    Running modified secant with initial guess 0.0500 ...
    >> The correct mass change rate: 5.8765e-02 kg/s



```python
rocket_solver = RocketSolver(rocket, dm=dm_300m)
t, model = rocket_solver.solve_model(heun_step)
ax = rocket_solver.plot(0)
ax.plot(t[-1], model[-1,0], '*', markersize=15, label=f'detonation at y = {model[-1,0]:.0f}')
ax.legend();
```


    
![png](Project_03_files/Project_03_33_0.png)
    

