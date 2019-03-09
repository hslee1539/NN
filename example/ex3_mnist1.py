import sys
sys.path.append(__file__.replace('\\NN\\example\\ex3_mnist1.py',''))

import NN as nn
import tensor
import pickle

network = nn.NetworkPlus(28*28)
network.append_affine(25).append_batchnormalization().append_relu()
network.append_affine(10).append_softmax()

optimizer = nn.optimizer.SGD()

with open(__file__.replace('ex3_mnist1.py', 'tensor_train_img.pkl'), 'rb') as f:
    train_img = pickle.load(f)
with open(__file__.replace('ex3_mnist1.py', 'tensor_train_label.pkl'), 'rb') as f:
    train_label = pickle.load(f)

train_img.shape = [train_img.shape[0], 28 * 28]

learner = nn.Learner(train_img, train_label, network, optimizer, 100)

learner.init()
learner.learn(10, True)

print(learner.findAccuracy())