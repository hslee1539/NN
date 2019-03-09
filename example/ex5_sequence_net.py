import sys
sys.path.append(__file__.replace('\\NN\\example\\ex5_sequence_net.py',''))

import NN as nn
import tensor

h = [1., 0., 0., 0.]
e = [0., 1., 0., 0.]
l = [0., 0., 1., 0.]
o = [0., 0., 0., 1.]

s1 = [1., 0., 0., 0., 0.]
s2 = [0., 1., 0., 0., 0.]
s3 = [0., 0., 1., 0., 0.]
s4 = [0., 0., 0., 1., 0.]
s5 = [0., 0., 0., 0., 1.]


x = tensor.Tensor()

cell = nn.NetworkPlus()