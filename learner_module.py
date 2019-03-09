import tensor
import NN as nn

class Learner:
    def __init__(self, data = nn.Dataset(tensor.Tensor([1],[1]), tensor.Tensor([1],[1])), network = nn.Network(), optimizer = nn.optimizer.SGD(0.01)):
        self.data = data
        self.network = network
        self.optimizer = optimizer

    def init(self, batch_size, max_index = 1):
        """네트워크를 배치에 맞게 초기화 합니다."""
        # 네트워크 내부 변수들 초기화용
        self.data.init(batch_size)
        self.network.initForward(self.data.x_batch)
        self.network.initBackward(1, self.data.y_batch)
        self.network.initPartial(max_index)
        return self

    def learn(self, epoch, isDisplayStatement = False):    
        for e in range(epoch):
            for i, x, y in self.data.normalRange():
                if(isDisplayStatement):
                    print('{0}/{1} epoch/epoch, {2}/{3} 1/epoch'.format(e, epoch, i, self.data.batch_count))
                self.network.forward()
                self.network.backward()
                self.network.update(self.optimizer)
    
    def partilLearn(self, epoch, index, isDisplayStatement = False):
        for e in range(epoch):
            for value in self.data.partialRange(index, self.network.max_index):
                left_loop = (epoch - e) * self.data.batch_count + value
                yield left_loop
                for value2 in self.network.partialForward(index):
                    yield left_loop + value2
                for value2 in self.network.partialBackward(index):
                    yield left_loop + value2
                for value2 in self.network.partialUpdate(self.optimizer, index):
                    yield left_loop + value2  
    
    def findAccuracy(self):
        acc = tensor.Tensor([0],[1])
        for i,x,y in self.data.normalRange():
            self.network.setX(x)
            result = self.network.forward()
            for j in range(len(result.array)):
                if(result.array[j] > 0.5):
                    acc.array[0] += y.array[j]
        acc.array[0] /= y.shape[0]
        return acc