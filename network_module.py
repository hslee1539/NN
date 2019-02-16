from .layer import interface_module
import tensor

class Network(interface_module.Forwardable, interface_module.Backwardable, interface_module.Learnable, interface_module.Updatable):
    def __init__(self):

        # 사용된 모든 레이어
        self.layers = []
        # 옵티마이저에 의한 업데이트가 필요한 레이어들
        self.updatable_layers = []
        # 순전파를 하는 레이어들(순서 중요)
        self.forwardable_layers = []
        # 역전파를 하는 레이어들(순서 중요)
        self.backwardable_layers = []
        # 테이블을 가지고 역전파를 할 수 있는 레이어들
        self.learnable_layers = []
        # 테이블을 가지고 역전파를 할 레이어
        self.learnable_layer = None

    def append(self, layer):
        self.layers.append(layer)
        for base in type(layer).__bases__:
            if(base == interface_module.Forwardable):
                self.forwardable_layers.append(layer)
                
            elif(base == interface_module.Backwardable):
                #self.backwardable_layers.append(layer)
                self.backwardable_layers = [layer] + self.backwardable_layers
                
            elif(base == interface_module.Updatable):
                self.updatable_layers.append(layer)

            elif(base == interface_module.Learnable):
                self.learnable_layers.append(layer)
                self.learnable_layer = layer
                
            else:
                print('Network warning: 지원하지 않는 인터페이스가 발견됬습니다.')
        return self

    def append_list(self, *layers):
        for layer in layers:
            self.append(layer)
        return self

    def forward(self, x):
        for layer in self.forwardable_layers:
            x = layer.forward(x)
            print(x) #debug
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

    def learn(self, t):
        dx = self.learnable_layer.learn(t)
        
        for layer in self.backwardable_layers:
            dx = layer.backward(dx)
        return dx

    def learn_line(self, t):
        dx = self.learnable_layer.learn_line(t)
        
        for layer in self.backwardable_layers:
            dx = layer.backward_line(dx)
        return dx

    def update(self, optimizer):
        for layer in self.updatable_layers:
            layer.update(optimizer)
        return None

    
