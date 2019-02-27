import tensor
import NN as nn

class Learner:
    def __init__(self, x, t, network, optimizer, batch_size):
        self.__x = x
        self.__t = t
        self.network = network
        self.optimizer = optimizer
        self.__batch_size = batch_size
        x_batch_shape = x.shape.copy()
        t_batch_shape = t.shape.copy()
        x_batch_shape[0] = batch_size
        t_batch_shape[0] = batch_size
        self.__x_batch = tensor.create_zeros(x_batch_shape)
        self.__t_batch = tensor.create_zeros(t_batch_shape)
        self.__all_index = tensor.create_arange(0, x.shape[0])
        self.__select_index = tensor.create_zeros([batch_size], int)

    def init(self):
        # 네트워크 내부 변수들 초기화용
        print(self.__x_batch.shape)
        self.network.forward(self.__x_batch)
        self.network.startBackward(self.__t_batch)

    def learn(self, epoch):
        # 학습 시작
        for e in range(epoch):
            print('epoch :' + str(e))
            tensor.set_shuffle(self.__all_index)
            
            for i in range(self.__x.shape[0] // self.__batch_size):
                tensor.copy(self.__all_index, i * self.__batch_size, self.__batch_size, self.__select_index)
                tensor.copy_row(self.__x, self.__select_index, self.__x_batch)
                tensor.copy_row(self.__t, self.__select_index, self.__t_batch)
                self.network.forward_line(self.__x_batch)
                self.network.startBackward_line(self.__t_batch)
                self.network.update(self.optimizer)
    
    def findAccuracy(self):
        acc = tensor.Tensor([0],[1])
        for i in range(self.__x.shape[0] // self.__batch_size):
            tensor.copy(self.__all_index, i * self.__batch_size, self.__batch_size, self.__select_index)
            tensor.copy_row(self.__x, self.__select_index, self.__x_batch)
            tensor.copy_row(self.__t, self.__select_index, self.__t_batch)

            result = self.network.forward_line(self.__x_batch)
            for j in range(len(self.__t_batch.array)):
                if(result.array[j] > 0.5):
                    acc.array[0] += self.__t_batch.array[j]
        acc.array[0] /= self.__t.shape[0]
        return acc
    