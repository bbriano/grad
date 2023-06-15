from grad import Perceptron
from math import exp

model = Perceptron(3, [4, 4, 1])
xs = [
	[2., 3., -1.],
	[3., -1., .5],
	[.5, 1., 1.],
	[1., 1., -1.],
]
ys = [1, -1, -1, 1]

N = 100
for k in range(N):
	outs = [model(x)[0] for x in xs]
	loss = sum((out-y)*(out-y) for out,y in zip(outs, ys))

	for p in model.parameters():
		p.grad = 0
	loss.backward()
	rate = 0.05
	for p in model.parameters():
		p.data -= p.grad * rate

	print("k={} loss={}".format(k, loss.data))

print("\npredicted:")
for o in outs:
	print(o.data)
