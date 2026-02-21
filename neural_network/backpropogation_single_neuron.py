import pandas as pd 
import numpy as np 
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    s=sigmoid(x)
    return s*(1-s) 
x=2.0
y=1.0
w=0.1
b=0.0
l_r=0.1
for i in range(10):
    z=w*x + b 
    # forward propogation
    y_hat=sigmoid(z)
    loss= (y-y_hat)**2
    # back propogate 
    dl_dyhat=-2*(y-y_hat)
    dyhat_dz=sigmoid_derivative(z)
    dz_dw=x
    dz_db=1
    dl_dw=dl_dyhat*dyhat_dz*dz_dw
    dl_db=dl_dyhat*dyhat_dz*dz_db
    
    #  weight update 
    w=w-l_r*dl_dw
    b=b-l_r*dl_db
    print(f" Step {i} loss : {loss:.4f}  w : {w:.2f} b : {b:.2f}")
print(f" Y_hat after backpropogation is : {sigmoid(w*x + b) }")
print(f" {0.42*2 + 0.16:.4f}")
    
