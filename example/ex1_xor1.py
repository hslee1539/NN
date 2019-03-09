import sys
sys.path.append(__file__.replace('\\NN\\example\\ex1_xor1.py',''))

import NN as nn

import tensor

# xor 데이터
x = tensor.Tensor([0., 0., 0., 1., 1., 0., 1., 1.], [4,2])
y = tensor.Tensor([0.,1.,1.,0.,1.,0.,0.,1.], [4,2])

# 변수 초기화
w1 = tensor.create_gauss([2,4])
b1 = tensor.create_zeros([4])
w2 = tensor.create_gauss([4,2])
b2 = tensor.create_zeros([2])

# 네트워크 생성
network = nn.Network()
network.append(nn.layer.Affine(w1, b1))
network.append(nn.layer.Sigmoid())
network.append(nn.layer.Affine(w2, b2))
network.append(nn.layer.Softmax())

optimizer = nn.optimizer.SGD(0.05)

# 초기화
network.initForward(x)
network.initBackward(1, y)

# 학습 되지 않은 상태에서 결과
print(network.forward())

# 학습
for i in range(10000):
    network.forward()
    network.backward()
    network.update(optimizer)

# 학습 후 결과
print(network.forward())