import sys
sys.path.append(__file__.replace('\\NN\\example\\ex1_xor1.py',''))
print(sys.path)

import NN as nn
import tensor

x = tensor.Tensor([0., 0., 0., 1., 1., 0., 1., 1.], [4,2])
y = tensor.Tensor([0.,1.,1.,0.,1.,0.,0.,1.], [4,2])

w1 = tensor.create_gauss([2,4])
b1 = tensor.create_zeros([4])
w2 = tensor.create_gauss([4,2])
b2 = tensor.create_zeros([2])


network = nn.Network()
network.append(nn.layer.Affine(w1, b1))
network.append(nn.layer.Relu())
network.append(nn.layer.Affine(w2, b2))
network.append(nn.layer.Softmax())

optimizer = nn.optimizer.SGD(0.05)

print('first out = ')
print(network.startForward(x))
network.startBackward(y)
network.update(optimizer)
for i in range(1000):
    network.startForward_line(x)
    network.startBackward_line(y)
    network.update(optimizer)

print(network.startForward_line(x))
