from . import interface_module
import tensor

class SGD(interface_module.Base):
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate

    def update(self, base, delta):
        for i in range(len(base.array)):
            base.array[i] -= self.learning_rate * delta.array[i]
        return None
    
