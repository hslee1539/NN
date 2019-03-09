import tensor
import NN as nn
import time

class Sync:
    def __init__(self, max_index):
        self.max_index = max_index
        self.state = [0] * max_index
    
    def iterator(self, iters, index):
        for item in iters:
            self.state[index] += 1
            while((sum(self.state) != self.max_index) & (self.state[index] != 0)):
                time.sleep(0.1)
            for i in range(self.max_index):
                self.state[i] = 0
            yield item
