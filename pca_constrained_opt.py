import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
data =  pd.read_csv('Boston.csv', index_col=0)
df2 = pd.DataFrame(data= data)
X = df2.loc[ : , ["crim","nox","rm","age","dis","tax","ptratio","lstat"]]
X = X.to_numpy()
a = np.mean(X, axis = 0)
X0 = X - a
def objective(v):
    projection = X0 @ v
    return -np.sum(projection**2)


def norm_constraint(v):
    return np.sum(v**2) - 1


m = 3  # По заданию требуется 3 вектора
basis = np.zeros((8, m))

for i in range(m):
    
    cons = [{'type': 'eq', 'fun': norm_constraint}]
    if i > 0:
        
        def ortho_constraint(v, current_i=i):
            return basis[:, :current_i].T @ v

        cons.append({'type': 'eq', 'fun': ortho_constraint})
    v_start = np.random.randn(8)
    res = minimize(objective, x0=v_start, method='SLSQP', constraints=cons)
    basis[:, i] = res.x
print("Stützvektor a:\n", a)
print("\nOrthonormalsystem v1, v2, v3:")
for i in range(m):
    print(f"v{i+1} =\n", basis[:, i])
  #results r Stützvektor a:
# [  3.61352356   0.55469506   6.28463439  68.57490119   3.79504269
 #408.23715415  18.4555336   12.65306324]

#Orthonormalsystem v1, v2, v3:
#v1 =
 #[-0.09172994 -0.03395989 -0.08219853  0.10190053 -0.11146938 -0.95107414
 # 0.23557624 -0.02878218]
#v2 =
 #[ 0.50213064  0.27419667  0.21855231 -0.43157587  0.08036681 -0.26205442
 #-0.55167684 -0.24326834]
#v3 =
# [-0.5292243  -0.27021922  0.044641   -0.49284286  0.43175222 -0.1209012
 #-0.25502745  0.36871429]
