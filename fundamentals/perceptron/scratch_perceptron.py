import numpy as np
class Perceptron:
    def __init__(self,lr=0.1,epochs=10):
        self.lr=lr
        self.epochs=epochs
    def fit(self,X,y):
        self.w=np.zeros(X.shape[1])
        self.b=0
        for _ in range(self.epochs):
            for  xi,yi in zip(X,y):
                # activation function
                z=np.dot(xi,self.w)+self.b
                # loss function
                loss=max(0,-yi*z)
                if loss>0:
                    self.w+=self.lr*yi*xi
                    self.b+=self.lr*yi
    def predict(self,X):
        
        return [1 if np.dot(xi,self.w)+self.b>=0 else -1 for xi in X]