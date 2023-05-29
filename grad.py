import random
import math

class Value:
	def __init__(self, data, children=[], op=""):
		self.data = data
		self.grad = 0
		self.chain = lambda: None
		self.children = children
		self.op = op

	def __add__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		out = Value(self.data + other.data, [self, other], "+")
		def chain():
			self.grad += out.grad
			other.grad += out.grad
		out.chain = chain
		return out

	def __mul__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		out = Value(self.data * other.data, [self, other], "*")
		def chain():
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad
		out.chain = chain
		return out

	def tanh(self):
		out = Value(math.tanh(self.data), [self], "tanh")
		def chain():
			self.grad += (1 - out.data*out.data) * out.grad
		out.chain = chain
		return out

	def backward(self):
		topo = []
		visited = set()
		def build(v):
			if id(v) in visited:
				return
			visited.add(id(v))
			for c in v.children:
				build(c)
			topo.append(v)
		build(self)
		self.grad = 1
		for v in reversed(topo):
			v.chain()

	def __neg__(self):
		return self * -1

	def __sub__(self, other):
		return self + -other

	def __rsub__(self, other):
		return self - other

	def __radd__(self, other):
		return self + other

	def __rmul__(self, other):
		return self * other

class Neuron:
	def __init__(self, nin):
		self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
		self.b = Value(0)

	def __call__(self, x):
		return (sum(wi*xi for wi,xi in zip(self.w, x)) + self.b).tanh()

	def parameters(self):
		return self.w + [self.b]

class Layer:
	def __init__(self, nin, nout):
		self.neurons = [Neuron(nin) for _ in range(nout)]

	def __call__(self, x):
		return [n(x) for n in self.neurons]

	def parameters(self):
		return [p for n in self.neurons for p in n.parameters()]

class Perceptron:
	def __init__(self, nin, nouts):
		size = [nin] + nouts
		self.layers = [Layer(size[i], size[i+1]) for i in range(len(nouts))]

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

	def parameters(self):
		return [p for l in self.layers for p in l.parameters()]
