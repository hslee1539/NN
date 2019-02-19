from .layer import interface_module
import tensor

class Network(interface_module.Forwardable, interface_module.Backwardable, interface_module.ForwardStartable, interface_module.BackwardStartable, interface_module.Updatable):
    def __init__(self):

        # 사용된 모든 레이어
        self.layers = []
        # 옵티마이저에 의한 업데이트가 필요한 레이어들
        self.updatable_layers = []
        # 순전파를 하는 레이어들(순서 중요)
        self.forwardable_layers = []
        # 역전파를 하는 레이어들(순서 중요)
        self.backwardable_layers = []

        # 테이블을 가지고 역전파를 할 레이어
        self.backwardStartable_layer = None

    def append(self, layer):
        self.layers.append(layer)
        for base in type(layer).__bases__:
            if(base == interface_module.Forwardable):
                self.forwardable_layers.append(layer)
                
            if(base == interface_module.Backwardable):
                self.backwardable_layers.append(layer)
                #self.backwardable_layers = [layer] + self.backwardable_layers
                
            if(base == interface_module.Updatable):
                self.updatable_layers.append(layer)

            if(base == interface_module.BackwardStartable):
                self.backwardStartable_layer = layer

        return self

    def append_list(self, *layers):
        for layer in layers:
            self.append(layer)
        return self

    def forward(self, x):
        for layer in self.forwardable_layers:
            x = layer.forward(x)
        return x

    def forward_line(self, x):
        for layer in self.forwardable_layers:
            x = layer.forward_line(x)
        return x

    def backward(self, dx):
        for layer in self.backwardable_layers:
            dx = layer.backward(dx)
        return dx

    def backward_line(self, dx):
        for layer in self.backwardable_layers:
            dx = layer.backward_line(dx)
        return dx

    def startForward(self, x):
        for layer in self.forwardable_layers:
            x = layer.forward(x)
        return x
    
    def startForward_line(self ,x):
        for layer in self.forwardable_layers:
            x = layer.forward_line(x)
        return x

    def startBackward(self, t):
        dx = self.backwardStartable_layer.startBackward(t)
        for i in range(len(self.backwardable_layers) -2, -1, -1):
            dx = self.backwardable_layers[i].backward(dx)

        return dx

    def startBackward_line(self, t):
        dx = self.backwardStartable_layer.startBackward_line(t)
        for i in range(len(self.backwardable_layers) -2, -1, -1):
            dx = self.backwardable_layers[i].backward_line(dx)

        return dx

    def update(self, optimizer):
        for layer in self.updatable_layers:
            layer.update(optimizer)
        return None

    
