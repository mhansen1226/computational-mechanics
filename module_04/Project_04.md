# Project 4


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu
plt.style.use('fivethirtyeight')
```


```python
fea_arrays = np.load('../projects/fea_arrays.npz')
K=fea_arrays['K'] * 1000
for line in K:
    print('[', end='')
    for k in line:
        print(f'{k:>6.1f}', end=' ')
    print(']')
```


```python
print(np.linalg.cond(K))
print(np.linalg.cond(K[2:13,2:13]))

print(f'Expected error in x=solve(K,b) is {10**(16-16)}')
print(f'Expected error in x=solve(K[2:13,2:13],b) is {10**(2-16)}')
```


```python
def solveLU(L,U,b):
    '''solveLU: solve for x when LUx = b
    x = solveLU(L,U,b): solves for x given the lower and upper 
    triangular matrix storage
    uses forward substitution for 
    1. Ly = b
    then backward substitution for
    2. Ux = y
    
    Arguments:
    ----------
    L = Lower triangular matrix
    U = Upper triangular matrix
    b = output vector
    
    returns:
    ---------
    x = solution of LUx=b '''
    n=len(b)
    x=np.zeros(n)
    y=np.zeros(n)
        
    # forward substitution
    for k in range(0,n):
        y[k] = b[k] - L[k,0:k]@y[0:k]
    # backward substitution
    for k in range(n-1,-1,-1):
        x[k] = (y[k] - U[k,k+1:n]@x[k+1:n])/U[k,k]
    return x
```


```python
class Truss:
    '''Class to store all necessary constants to solve for truss deformation
    
    Attributes:
    -----------
    material    : string containing the material the truss is made from
    E           : young's modulus in MPa
    A           : cross sectional area in mm^2
    p           : density in g/mm^3
    isSolved    : ensures that a solution is completed before displaying results
    K           : stiffness matrix
    nodes       : list of node positons
    elems       : list of elements in the truss
    l           : length of each element
    F           : force array for solving the matix equation
    u           : deflection array for solving the matrix equation
    deformation : array with equivalent data to u, reshaped to separate x and y deflection
    forces      : array with equivalent data to F, reshaped to separate x and y forces
    max_dx      : stores maximum node deflection in x direction
    max_dy      : stores maximum node deflection in y direction
    
    Methods:
    --------
    
    __init__(material='Steel', E=200e3, A=0.1, p=7.85e-3):
        - store material, E, A, p, and set isSolved = False and create a Truss object
        
    __str__():
        - formats the output of calling print on the truss as shown below:
            Aluminum Truss Structure:
                7 nodes, 11 elements
                Young's Modulus: 70000.0 MPa
                Cross Sectional Area: 0.100 mm^2
                Memeber Length: 300.0 mm
                Density: 0.00785 g/mm^3
                Total Mass: 2.590 g
                Max defleciton:	x: 18.558 mm
                                y: 46.071 mm
    
    load_structure(fea):
        - load in K, nodes, elems from the saved npz file
        - calculate l and initialize F and u to all zeros
    
    plot(show_cstr=True, label_nodes=False, label_elems=False, ax=None):
        - plot the truss structure on ax (if ax is not provided, creates a default axes)
        - boolean options control whether aditional information is shown
    
    plot_deformation(def_scale=5):
        - first calls plot, then overlays the deformed structure according to the scale factor
        - shows force and displacement vectors as well (not scaled)

    apply_force(node, F):
        - apply 2D force F on specified node (stores in F array)

    solve():
        - solves the matrix equation with LU decomposition
        - stores deformation, forces, max_dx, max_dy, and sets isSolved = True

    print_solution():
        - displays the calculated deflections and forces at each node if isSolved is True
    
    '''
    def __init__(self, material='Steel', E=200e3, A=0.1, p=7.85e-3):
        self.material = material.capitalize()
        self.E = E 
        self.A = A 
        self.p = p
        self.isSolved = False
    
    def __str__(self):
        s = f'{self.material} Truss Structure:\n'
        s += f'\t{len(self.nodes)} nodes, {len(self.elems)} elements\n'
        s += f"\tYoung's Modulus: {self.E} MPa\n"
        s += f'\tCross Sectional Area: {self.A:.3f} mm^2\n'
        s += f'\tMemeber Length: {self.l} mm\n'
        s += f'\tDensity: {self.p} g/mm^3\n'
        s += f'\tTotal Mass: {len(self.elems)*self.l*self.A*self.p:.3f} g\n'
        s += F'\tMax defleciton:\tx: {self.max_dx:.3f} mm\n\t\t\ty: {self.max_dy:.3f} mm'
        return s
        
    def load_structure(self, fea):
        self.K = fea['K']
        self.nodes = fea['nodes']
        self.elems = fea['elems']
        self.l = self.nodes[2,1] - self.nodes[0,1]
        self.F = np.zeros(len(self.K))
        self.u = np.zeros(len(self.K))
    
    def plot(self, show_cstr=True, label_nodes=False, label_elems=False, ax=None):
        if ax is None:
            ax = plt.axes(title=f'{self.material} Truss Structure', 
                      xlabel='x (mm)', ylabel='y (mm)')
        
        if show_cstr:
            ax.set_title(ax.get_title() + ':\ntriangle markers are constraints')
            csrt_offset = 20
            ax.plot(self.nodes[0, 1], self.nodes[0,2]-csrt_offset, '^', color='r', markersize=15)
            ax.plot(self.nodes[0, 1]-csrt_offset, self.nodes[0,2], '>', color='k', markersize=15)
            ax.plot(self.nodes[-1, 1], self.nodes[-1, 2]-csrt_offset, '^', color='r', markersize=15)
        
        node_order = np.array([2, 1, 3, 5, 7, 6, 4, 2, 3, 4, 5, 6]) - 1
        node_path = self.nodes[node_order, 1:]
        
        ax.plot(node_path[:,0], node_path[:,1], '-', color='k')
        ax.plot(self.nodes[:, 1], self.nodes[:, 2], 'o', color='b')
        
        if label_nodes:
            for n in self.nodes:
                if n[2] > 0.8*self.l: offset = self.l/10
                else: offset = -self.l/5
                ax.text(n[1]-self.l/3, n[2]+offset, f'n {int(n[0])}', color='b')

        if label_elems:
            for e in self.elems:
                n1 = self.nodes[e[1]-1]
                n2 = self.nodes[e[2]-1]
                x = np.mean([n2[1], n1[1]])
                y = np.mean([n2[2], n1[2]])
                ax.text(x-self.l/5, y-self.l/10, f'el {int(e[0])}', color='r', bbox={'facecolor': 'lightgrey','alpha': 0.8})
        ax.axis(self.l*np.array([-0.5, 3.5, -1, 1.5]))
        return ax

    def plot_deformation(self, def_scale=5):
        ax = self.plot(show_cstr=False)
        ax.set_title(f'{self.material} Truss Deformation\n (Displacement scale: x{def_scale})')
        node_order = np.array([2, 1, 3, 5, 7, 6, 4, 2, 3, 4, 5, 6]) - 1
        node_path = self.nodes[node_order, 1:] + self.deformation[node_order]*def_scale
        plt.quiver(self.nodes[:,1], self.nodes[:,2], self.forces[:,0], self.forces[:,1], color='r', label='applied forces')
        plt.quiver(self.nodes[:,1], self.nodes[:,2], self.deformation[:,0], self.deformation[:,1], color='b', label=f'displacements')
        ax.plot(node_path[:,0], node_path[:,1], '-', color='red', linewidth=2, alpha=0.7)
        ax.legend()
        return ax
        
    def apply_force(self, node, F):
        self.F[(node-1)*2:node*2] = F
        return self.F
    
    def solve(self):
        P, L, U = lu(self.E * self.A * self.K[2:-1, 2:-1])
        self.u[2:-1] = solveLU(L, U, self.F[2:-1])
        self.F = self.K*self.E*self.A @ self.u
        
        self.deformation = self.u.reshape(len(self.nodes), 2)
        self.forces = self.F.reshape(len(self.nodes), 2)
        self.max_dx = max(np.abs(self.deformation[:,0]))
        self.max_dy = max(np.abs(self.deformation[:,1]))
        self.isSolved = True
    
    def print_solution(self):
        if self.isSolved:
            xy={0:'x',1:'y'}
            print('Displacements:\n----------------')
            for i, u in enumerate(self.deformation):
                print(f'u_{i}: {u[0]:>8.2f}i + {u[1]:>8.2f}j mm')
            print('\nForces:\n----------------')
            for i, F in enumerate(self.forces):
                print(f'F_{i}: {F[0]:>8.2f}i + {F[1]:>8.2f}j N')            
        else:
            raise Exception("Error: The reaction forces have not been solved")
```


```python
steel = Truss('steel', 200e3)
aluminum = Truss('aluminum', 70e3)
for truss in steel, aluminum:
    truss.load_structure(fea_arrays)
    truss.apply_force(node=4, F=(0, -300))
    truss.solve()
```


```python
steel.plot();
```


```python
steel.print_solution()
steel.plot_deformation();
```


```python
aluminum.print_solution()
aluminum.plot_deformation(def_scale=1);
```


```python
dA = 0.001
for truss in steel, aluminum:
    while truss.max_dy > 2:
        truss.A += dA
        truss.solve()
    print(truss, '\n')
```

```python

```
