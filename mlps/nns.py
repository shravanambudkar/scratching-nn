from typing import Any
from backproping import Value
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

class Neuron:
    
    def __init__(self, len_ins):
        self.len_ins = len_ins
        self.weights = [Value(np.random.rand()) for _ in range(len_ins)]
        self.bias = Value(np.random.rand())
        
    def parameters(self):
        return self.weights + [self.bias]
        
    def __call__(self, x):
        assert len(x) == self.len_ins, "Inputs and established lengths of inputs should match"
        out = sum((wi*xi for wi,xi in zip(self.weights, x)), self.bias)
        
        return out.tanh()
    
    
class Layer:
    
    def __init__(self, nins, nouts):
        self.neurons = [Neuron(nins) for _ in range(nouts)]
        
    def parameters(self):
        return [x for neuron in self.neurons for x in neuron.parameters()]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        out = out[0] if len(out)==1 else out
        return out
    
class MLP:
    def __init__(self, inputs, layers):
        nl = [inputs] + layers
        self.mlp = [Layer(nl[i],nl[i+1]) for i in range(len(layers))]
        
    def parameters(self):
        return [x for layer in self.mlp for x in layer.parameters()]
            
    def __call__(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x
    
    def _update_params(self,params, lr):
        for p in params:
            p.data = p.data - lr*p.grad
    
    def fit(self,X,Y,lr=0.03,epochs=10):
        for _ in range(epochs):
            out = [self.__call__(x) for x in X]
            loss = sum((out - Y)**2)
            print(f'Loss at epoch: {_ + 1} is {loss.data}')
            for p in self.parameters():
                p.grad = 0.0
            loss.backprop()
            self._update_params(self.parameters(),lr)
            
                
        
    
if __name__ == '__main__':
    mpls = MLP(3,[2,1])
    X, Y = make_regression(n_samples=3, n_features=3, noise=10, random_state=10)
    Y = MinMaxScaler().fit_transform(Y.reshape(-1, 1)).reshape(-1)

    mpls.fit(X,Y,epochs=20)
    # out = [mpls(xx) for xx in X]
    # # print(mpls.parameters())
    # loss = sum((pred - act)**2 for pred,act in zip(out,Y))
    # print(loss)
    # loss.backprop()