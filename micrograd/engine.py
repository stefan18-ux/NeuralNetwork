from draw import draw_dot
import math
class Value:
    """Stores a single scalar and it's gradient"""
    # Consider L = loss function

    def __init__(self, data, _children = "", _op = "", label = ""):
        self.data = data
        self.grad = 0
        self._backward = lambda:None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
       return f"Value(data = {self.data}, grad = {self.grad})\n"

    def __add__(self, other): # c = a + b
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += out.grad * 1 # (dL/dc * dc/da) where dL/dc is c.grad and dd/da is 1
            other.grad += out.grad * 1 # (dL/dc * dc/da) where dL/dc is c.grad and dd/db is 1
        
        out._backward = backward
        return out
    
    def __mul__(self, other): # c = a * b
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += out.grad * other.data # (dL/dc * dc/da) where dL/dc is c.grad and dd/da is b
            other.grad += out.grad * self.data # (dL/dc * dc/da) where dL/dc is c.grad and dd/db is a
        
        out._backward = backward
        return out
    
    def __pow__(self, other): # c = a ^ (int)
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self, ), f"**{other}")

        def backward():
            self.grad += (self.data**(other - 1) * other) * out.grad # (dL/dc * dc/da) where dL/dc is c.grad and dd/da is (int) * a^(int - 1)
        
        out._backward = backward
        return out
    
    def exp(self): # c = e ^ a
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad # (dL/dc * dc/da) where dL/dc is c.grad and dd/da is e ^ a which is math.exp(a) or out.data
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other): # c = b * a
        return self * other

    def __truediv__(self, other): # c = a / b
        return self * other**-1

    def __neg__(self): # c = -a
        return self * -1

    def __sub__(self, other): # c = a - b
        return self + (-other)

    # I have no fucking idea what this is
    def __radd__(self, other): # other + self
        return self + other
    
    # Activation function
    def tanh(self): # c = tanh(a)
        x = self.data
        uped = math.exp(2 * x)
        out = Value((uped - 1) / (uped + 1), (self, ), "tanh")

        def backward():
            self.grad += out.grad * (1 - out.data**2) # (dL/dc * dc/da) where dL/dc is c.grad and dd/da is 1 - tanh^2(a)
    
        out._backward = backward
        return out

    def backward(self):
        topo = []
        visited = set()
        
        def topological_sort(node):
            visited.add(node)
            for children in node._prev:
                if children not in visited:
                    topological_sort(children)
            topo.append(node)
        topological_sort(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()
        
                    

# x1 = Value(2.0, label='x1')
# x2 = Value(0.0, label='x2')
# # weights w1,w2
# w1 = Value(-3.0, label='w1')
# w2 = Value(1.0, label='w2')
# # bias of the neuron
# b = Value(6.8813735870195432, label='b')
# # x1*w1 + x2*w2 + b
# x1w1 = x1*w1; x1w1.label = 'x1*w1'
# x2w2 = x2*w2; x2w2.label = 'x2*w2'
# x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
# n = x1w1x2w2 + b; n.label = 'n'
# # ----
# # e = (2*n).exp()
# # o = (e - 1) / (e + 1)
# # ----
# o = n.tanh()
# o.label = 'o'
# o.backward()
# draw_dot(o)
