import numpy as np
import math


class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
    
    def __repr__(self) -> str:
        return f'Value(data={self.data}, Child = {self._prev} , grad = {self.grad})'
    
    def __add__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(data = self.data + other.data, _children=(self, other))
        
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out
    
    def __pow__(self,other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(data = self.data ** other.data, _children=(self, other))
        def backward():
            self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
            other.grad += (out.data * math.log(self.data)) * out.grad
        out._backward = backward
        return out
    
    def __mul__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(data = self.data * other.data, _children=(self, other))
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out
    
    def __sub__(self,other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(data = self.data - other.data, _children=(self, other))
        def backward():
            self.grad += 1 * out.grad
            other.grad += -1 * out.grad
        out._backward = backward
        return out
    
    def __truediv__(self,other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(data = self.data * (other.data**-1), _children=(self, other))
        def backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += -1 * (self.data / other.data**2) * out.grad
        out._backward = backward
        return out
    
    def tanh(self):
        out = Value(data=(math.exp(2*self.data)-1)/(math.exp(2*self.data)+1), _children = (self,))
        def backward():
            self.grad += (1 - (out.data ** 2)) * out.grad
        out._backward = backward
        return out
    
    def sigmoid(self):
        out = Value(data = 1/(1+math.exp(-1*self.data)), _children = (self,))
        def backward():
            self.grad += (out.data * (1-out.data)) * out.grad
        out._backward = backward
        return out
    
    def backprop(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    
        

if __name__ == '__main__':        
    a = Value(3.0)
    b = Value(2.0)
    c = a*b
    d = c.sigmoid()
    c.backprop()
    print(c)